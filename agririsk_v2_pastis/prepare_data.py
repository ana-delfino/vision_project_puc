"""
Preparo de dados · AgriRisk-MA v2 (trigo · PASTIS-HD)

    python prepare_data.py                         # 30 recortes por fold (150), ~3 GB baixados
    python prepare_data.py --folds 5 --per-fold 6  # só alguns recortes de teste (demo local)

Seleção (volumetria mínima, sem viés de região)
  - Só recortes com ≥5% de pixels de trigo (índice em data_index/wheat_patches.json,
    gerado a partir das anotações oficiais).
  - Em cada fold, as regiões (tiles) entram em rodízio → todas representadas.

Pré-processamento por recorte (o arquivo bruto é apagado depois)
  - Sentinel-2 (38-61 datas irregulares, com nuvens) → máscara de nuvem (azul alto)
    → composição mediana numa grade fixa de 26 passos de 16 dias → lacunas
    preenchidas por interpolação linear no tempo. Resultado: (26, 10, 128, 128) float16.
  - SPOT 1 m → 2 m (640 x 640 RGB).
  - Rótulos (cultura por pixel) e ids de parcela, a 10 m.
  - Curva média de cada parcela (26 x 10), usada pelo baseline e pelo benchmark regional.
"""
import argparse
import json
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import cv2
import numpy as np

import config as C

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
warnings.filterwarnings("ignore", message="All-NaN slice")   # pixel sem dado limpo na janela
RAW = C.DATA / "raw"
PATCHES = C.DATA / "patches"


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------
def select_patches(folds, per_fold):
    idx = json.loads((C.INDEX / "wheat_patches.json").read_text())["patches"]
    chosen = []
    for f in folds:
        # gerador POR FOLD: a seleção de um fold não depende de quais outros folds foram pedidos
        rng = np.random.default_rng(C.SEED + f)
        by_tile = {}
        for r in idx:
            if r["fold"] == f and r["wheat_frac"] >= C.MIN_WHEAT_FRAC:
                by_tile.setdefault(r["tile"], []).append(r)
        for rows in by_tile.values():
            rng.shuffle(rows)
        tiles, picked = sorted(by_tile), []
        while len(picked) < per_fold and any(by_tile.values()):
            for t in tiles:                                    # rodízio entre regiões
                if by_tile[t] and len(picked) < per_fold:
                    picked.append(by_tile[t].pop())
        chosen += picked
    return chosen


def load_metadata():
    path = C.DATA / "metadata.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(C.METADATA_URL, path)
    feats = json.loads(path.read_text())["features"]
    out = {}
    for f in feats:
        p = f["properties"]
        d = p["dates-S2"]
        d = d if isinstance(d, dict) else json.loads(d.replace("'", '"'))
        out[int(p["ID_PATCH"])] = [date(int(str(v)[:4]), int(str(v)[4:6]), int(str(v)[6:])) for v in
                                   (d[k] for k in sorted(d, key=int))]
    return out


# ---------------------------------------------------------------------------
# Série temporal: nuvens → composição em grade fixa → interpolação
# ---------------------------------------------------------------------------
def fill_nan_time(a: np.ndarray) -> np.ndarray:
    """Interpolação linear ao longo do eixo 0 (tempo), vetorizada por pixel."""
    T = a.shape[0]
    flat = a.reshape(T, -1)
    v = ~np.isnan(flat)
    t = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(v, t, -1), axis=0)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(v, t, T), 0), axis=0), 0)
    cols = np.broadcast_to(np.arange(flat.shape[1])[None], flat.shape)
    pv = flat[np.clip(prev, 0, T - 1), cols]
    nv = flat[np.clip(nxt, 0, T - 1), cols]
    w = (t - prev) / np.maximum(nxt - prev, 1)
    has_p, has_n = prev >= 0, nxt < T
    out = np.where(v, flat, np.where(has_p & has_n, pv + w * (nv - pv),
                                     np.where(has_p, pv, np.where(has_n, nv, 0.0))))
    return out.reshape(a.shape)


def temporal_composite(s2: np.ndarray, dates: list) -> tuple[np.ndarray, float]:
    days = np.array([(d - C.T_START).days for d in dates])
    x = s2.astype(np.float32) / 1e4
    blue = s2[:, C.B_BLUE]
    valid = (blue < C.CLOUD_BLUE_THRESHOLD) & (blue > 0)
    x[np.broadcast_to(~valid[:, None], x.shape)] = np.nan
    out = np.full((C.T_STEPS,) + x.shape[1:], np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # "All-NaN slice"
        for k in range(C.T_STEPS):
            sel = np.abs(days - k * C.T_STEP_DAYS) <= C.T_STEP_DAYS
            if sel.any():
                out[k] = np.nanmedian(x[sel], axis=0)
    filled_frac = float(np.isnan(out).mean())
    return fill_nan_time(out).astype(np.float16), filled_frac


def parcel_curves(comp: np.ndarray, ids: np.ndarray, target: np.ndarray):
    """Curva média (T, 10) e cultura majoritária de cada parcela."""
    flat_ids = ids.ravel()
    uniq = np.unique(flat_ids)
    uniq = uniq[uniq != 0]
    inv = np.searchsorted(uniq, flat_ids)
    inside = flat_ids != 0
    cnt = np.bincount(inv[inside], minlength=len(uniq)).astype(np.float32)
    T, B = comp.shape[:2]
    curves = np.zeros((len(uniq), T, B), np.float32)
    c32 = comp.astype(np.float32).reshape(T, B, -1)
    for t in range(T):
        for b in range(B):
            curves[:, t, b] = np.bincount(inv[inside], weights=c32[t, b][inside], minlength=len(uniq)) / cnt
    labels = np.zeros(len(uniq), np.int64)
    tflat = target.ravel()
    for i, p in enumerate(uniq):
        labels[i] = np.bincount(tflat[flat_ids == p], minlength=C.N_CLASSES).argmax()
    return uniq, labels, cnt.astype(np.int32), curves.astype(np.float16)


# ---------------------------------------------------------------------------
# Download + processamento de um recorte
# ---------------------------------------------------------------------------
def _get(url, dst, attempts=4):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def process_patch(row, dates):
    p = row["patch"]
    out = PATCHES / f"{p}.npz"
    if out.exists():
        return p, "cache"
    RAW.mkdir(parents=True, exist_ok=True)
    files = {k: RAW / f"{k}_{p}{'.tif' if k == 'spot' else '.npy'}" for k in C.HF_FILES}
    for k, path in files.items():
        if not path.exists():
            _get(C.HF_BASE + C.HF_FILES[k].format(p=p), path)

    s2 = np.load(files["s2"])
    comp, gap = temporal_composite(s2, dates[p])
    spot = cv2.cvtColor(cv2.imread(str(files["spot"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    spot = cv2.resize(spot, (C.SPOT_PX, C.SPOT_PX), interpolation=cv2.INTER_AREA)
    target = np.load(files["target"])[0].astype(np.uint8)
    ids = np.load(files["parcels"]).astype(np.int32)
    pids, plabels, pnpix, pcurves = parcel_curves(comp, ids, target)

    PATCHES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, s2=comp, spot=spot, target=target, parcels=ids,
                        parcel_ids=pids, parcel_labels=plabels, parcel_npix=pnpix,
                        parcel_curves=pcurves, tile=row["tile"], fold=row["fold"], gap_frac=gap)
    for path in files.values():
        path.unlink(missing_ok=True)                     # descarta o bruto
    return p, f"ok (lacunas temporais preenchidas: {gap * 100:.0f}%)"


def main(folds, per_fold, workers=8):
    t0 = time.time()
    dates = load_metadata()
    rows = select_patches(folds, per_fold)
    print(f"[prepare] {len(rows)} recortes · folds {folds} · {per_fold} por fold")
    with ThreadPoolExecutor(workers) as ex:
        for i, (p, status) in enumerate(ex.map(lambda r: process_patch(r, dates), rows), 1):
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} · recorte {p}: {status} · {time.time() - t0:.0f}s")
    manifest = [{"patch": r["patch"], "tile": r["tile"], "fold": r["fold"], "wheat_frac": r["wheat_frac"]}
                for r in rows if (PATCHES / f"{r['patch']}.npz").exists()]
    # mescla com o manifesto existente (ex.: rodar só o fold de teste depois)
    mpath = C.DATA / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else []
    merged = {m["patch"]: m for m in old + manifest}
    mpath.write_text(json.dumps(sorted(merged.values(), key=lambda m: m["patch"]), indent=0))
    mb = sum(f.stat().st_size for f in PATCHES.glob("*.npz")) / 1e6
    print(f"[prepare] pronto: {len(merged)} recortes no manifesto · {mb:.0f} MB em {PATCHES} · "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--per-fold", type=int, default=C.PATCHES_PER_FOLD)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.folds, a.per_fold, a.workers)
