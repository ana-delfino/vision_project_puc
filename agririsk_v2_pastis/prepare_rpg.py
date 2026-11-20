"""
Polígonos declarados (RPG 2019 · IGN/ASP) → rótulos de delimitação na grade do SPOT

    python prepare_rpg.py extract     # (local) .7z regionais → data_index/rpg_polygons.json.gz
    python prepare_rpg.py rasterize   # (local ou VM) rótulos a 2 m para os recortes do manifesto
    python prepare_rpg.py check       # alinhamento: RPG rasterizado a 10 m × ParcelIDs do PASTIS

O RPG é a declaração anual dos produtores à PAC (Política Agrícola Comum): polígono da
parcela, cultura declarada (CODE_CULTU/CODE_GROUP) e área declarada (SURF_PARC, ha).
O ID_PARCEL do RPG é o mesmo id de parcela do PASTIS → liga declaração e imagens.

Georreferenciamento: cada recorte do PASTIS é um quadrado na grade UTM do Sentinel-2,
levemente rotacionado em Lambert-93 (EPSG:2154, o sistema do RPG). A transformação
Lambert-93 → pixel é estimada por mínimos quadrados a partir dos 4 cantos (erro
submétrico num quadrado de 1,28 km).
"""
import gzip
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import config as C

RPG_DIR = C.DATA / "rpg"
POLY_FILE = C.INDEX / "rpg_polygons.json.gz"
LABEL_DIR = C.DATA / "rpg_labels"
REGIONS = ["R28", "R93", "R76", "R27", "R84", "R44"]   # regiões que cobrem os 4 tiles
RPG_URL = ("https://data.geopf.fr/telechargement/download/RPG/RPG_2-0_SHP_LAMB93_{r}-2019/"
           "RPG_2-0_SHP_LAMB93_{r}-2019.7z")


def patch_corners():
    feats = json.loads((C.DATA / "metadata.geojson").read_text())["features"]
    out = {}
    for f in feats:
        c = f["geometry"]["coordinates"]
        while isinstance(c[0][0], list):
            c = c[0]
        out[int(f["properties"]["ID_PATCH"])] = np.array(c[:4], np.float64)
    return out


def affine_to_pixel(corners: np.ndarray, size: int) -> np.ndarray:
    """Matriz 2x3 que leva (x, y) Lambert-93 → (col, lin) em pixels da grade `size`."""
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64)
    A = np.c_[corners, np.ones(4)]
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)                  # (3, 2)
    return M.T


def selected_patches():
    from prepare_data import select_patches
    return [r["patch"] for r in select_patches([1, 2, 3, 4, 5], C.PATCHES_PER_FOLD)]


# ---------------------------------------------------------------------------
# 1. EXTRACT: só os polígonos que tocam os recortes selecionados
# ---------------------------------------------------------------------------
def extract():
    import py7zr
    import shapefile
    import urllib.request
    corners = patch_corners()
    sel = selected_patches()
    boxes = {p: (corners[p][:, 0].min(), corners[p][:, 1].min(), corners[p][:, 0].max(), corners[p][:, 1].max())
             for p in sel}
    allx0 = min(b[0] for b in boxes.values()); ally0 = min(b[1] for b in boxes.values())
    allx1 = max(b[2] for b in boxes.values()); ally1 = max(b[3] for b in boxes.values())
    out = {str(p): [] for p in sel}
    RPG_DIR.mkdir(parents=True, exist_ok=True)
    for r in REGIONS:
        arch = RPG_DIR / f"{r}.7z"
        if not arch.exists():
            print(f"[rpg] baixando {r} ...")
            urllib.request.urlretrieve(RPG_URL.format(r=r), arch)
        tmp = RPG_DIR / f"x_{r}"
        with py7zr.SevenZipFile(arch) as z:
            z.extract(path=tmp, targets=[n for n in z.getnames() if "PARCELLES_GRAPHIQUES" in n])
        shp = next(tmp.rglob("PARCELLES_GRAPHIQUES.shp"))
        rd = shapefile.Reader(str(shp))
        n = 0
        for sr in rd.iterShapeRecords():
            x0, y0, x1, y1 = sr.shape.bbox
            if x1 < allx0 or x0 > allx1 or y1 < ally0 or y0 > ally1:
                continue
            hits = [p for p, b in boxes.items() if not (x1 < b[0] or x0 > b[2] or y1 < b[1] or y0 > b[3])]
            if not hits:
                continue
            rec = sr.record.as_dict()
            pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
            rings = [[[round(x, 2), round(y, 2)] for x, y in pts[parts[i]:parts[i + 1]]] for i in range(len(parts) - 1)]
            item = {"id": int(rec["ID_PARCEL"]), "surf_ha": float(rec["SURF_PARC"]),
                    "group": int(rec["CODE_GROUP"]) if str(rec["CODE_GROUP"]).isdigit() else -1,
                    "cultu": rec["CODE_CULTU"], "rings": rings}
            for p in hits:
                out[str(p)].append(item)
            n += 1
        print(f"[rpg] {r}: {n} polígonos nos recortes selecionados")
        shutil.rmtree(tmp)
    with gzip.open(POLY_FILE, "wt") as f:
        json.dump(out, f)
    print(f"[rpg] {sum(len(v) for v in out.values())} polígonos → {POLY_FILE} "
          f"({POLY_FILE.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2. RASTERIZE
# ---------------------------------------------------------------------------
_POLY = None


def polygons(patch):
    global _POLY
    if _POLY is None:
        with gzip.open(POLY_FILE, "rt") as f:
            _POLY = json.load(f)
    return _POLY.get(str(patch), [])


def rasterize_ids(patch, size, corners=None):
    """Mapa de ids de parcela RPG (int32) numa grade size x size do recorte."""
    corners = corners if corners is not None else patch_corners()[patch]
    M = affine_to_pixel(corners, size)
    ids = np.zeros((size, size), np.int32)
    # maiores primeiro, para que parcelas pequenas não sejam apagadas por sobreposição
    for poly in sorted(polygons(patch), key=lambda q: -q["surf_ha"]):
        rings = []
        for ring in poly["rings"]:
            xy = np.asarray(ring, np.float64)
            px = (np.c_[xy, np.ones(len(xy))] @ M.T)
            rings.append(np.round(px * 8).astype(np.int32))        # precisão sub-pixel (shift=3)
        cv2.fillPoly(ids, rings, int(poly["id"]), lineType=cv2.LINE_8, shift=3)
    return ids


def boundaries(ids: np.ndarray, width_px=1) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    mx = cv2.dilate(ids.astype(np.float32), k)
    mn = cv2.erode(ids.astype(np.float32), k)
    b = (mx != mn)
    return cv2.dilate(b.astype(np.uint8), np.ones((2 * width_px - 1,) * 2, np.uint8)).astype(bool) if width_px > 1 else b


def rasterize():
    corners = patch_corners()
    man = json.loads((C.DATA / "manifest.json").read_text())
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for m in man:
        p = m["patch"]
        ids = rasterize_ids(p, C.SPOT_PX, corners[p])
        meta = {q["id"]: (q["surf_ha"], q["group"], q["cultu"]) for q in polygons(p)}
        keys = np.array(sorted(meta), np.int64)
        np.savez_compressed(LABEL_DIR / f"{p}.npz", ids=ids, poly_ids=keys,
                            surf_ha=np.array([meta[k][0] for k in keys], np.float32),
                            group=np.array([meta[k][1] for k in keys], np.int32),
                            cultu=np.array([meta[k][2] for k in keys]))
    print(f"[rpg] rótulos a {C.SPOT_GSD_M:.0f} m para {len(man)} recortes → {LABEL_DIR}")


def check():
    """IoU por parcela entre o RPG rasterizado a 10 m e os ParcelIDs oficiais do PASTIS."""
    from common import load_patch
    corners = patch_corners()
    man = json.loads((C.DATA / "manifest.json").read_text())
    ious = []
    for m in man[:20]:
        d = load_patch(m["patch"])
        rp = rasterize_ids(m["patch"], 128, corners[m["patch"]])
        for pid in d["parcel_ids"][:40]:
            a, b = d["parcels"] == pid, rp == pid
            if a.sum() >= 20:
                ious.append((a & b).sum() / max((a | b).sum(), 1))
    ious = np.array(ious)
    print(f"[rpg] alinhamento em {len(ious)} parcelas: IoU mediana {np.median(ious):.3f} · "
          f"p10 {np.percentile(ious, 10):.3f} · {np.mean(ious > 0.7) * 100:.0f}% com IoU > 0,7")


if __name__ == "__main__":
    {"extract": extract, "rasterize": rasterize, "check": check}[sys.argv[1]]()
