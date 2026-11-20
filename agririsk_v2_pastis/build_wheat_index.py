"""
Gera data_index/wheat_patches.json: quais recortes do PASTIS-HD têm trigo de inverno.

    python build_wheat_index.py               # baixa só as anotações (~5 mil arquivos pequenos, ~500 MB)
    python build_wheat_index.py --check       # regenera e compara com o índice commitado

Por recorte (2.433 no PASTIS-HD) grava, se houver pixel de trigo (classe 2):
  patch, tile (região Sentinel-2), fold oficial, wheat_frac (fração de pixels de trigo),
  wheat_parcels (nº de parcelas com trigo) e n_dates_s2 (nº de imagens Sentinel-2).

O índice existe para que prepare_data.py escolha os recortes sem baixar todas as
anotações a cada execução. As anotações ficam em cache em data/annotations/.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as C

ANN_DIR = C.DATA / "annotations"
OUT = C.INDEX / "wheat_patches.json"


def _get(url, dst, attempts=5):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))             # o Hugging Face limita a taxa de pedidos


def download_annotations(patches, workers=16):
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(C.HF_BASE + C.HF_FILES[k].format(p=p), ANN_DIR / f"{k}_{p}.npy")
            for p in patches for k in ("target", "parcels")]
    todo = [j for j in jobs if not j[1].exists()]
    print(f"[index] {len(jobs) - len(todo)} anotações em cache · baixando {len(todo)}")
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(lambda j: _get(*j), todo), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}")


def build():
    from prepare_data import load_metadata
    load_metadata()                                          # garante data/metadata.geojson
    feats = json.loads((C.DATA / "metadata.geojson").read_text())["features"]
    meta = [f["properties"] for f in feats]
    download_annotations([int(m["ID_PATCH"]) for m in meta])
    rows = []
    for m in meta:                                           # mesma ordem do metadata.geojson
        p = int(m["ID_PATCH"])
        t = np.load(ANN_DIR / f"target_{p}.npy")[0]
        ids = np.load(ANN_DIR / f"parcels_{p}.npy")
        wheat = t == C.CROP_ID
        if not wheat.any():
            continue
        d = m["dates-S2"]
        d = d if isinstance(d, dict) else json.loads(d.replace("'", '"'))
        rows.append({"patch": p, "tile": m["TILE"], "fold": int(m["Fold"]),
                     "wheat_frac": round(float(wheat.mean()), 4),
                     "wheat_parcels": len({int(q) for q in np.unique(ids[wheat]) if q}),
                     "n_dates_s2": len(d)})
    return {"source": "PASTIS-HD ANNOTATIONS/TARGET_*.npy (classe 2 = Soft winter wheat)", "patches": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compara com o índice commitado sem sobrescrever")
    a = ap.parse_args()
    idx = build()
    n5 = sum(r["wheat_frac"] >= C.MIN_WHEAT_FRAC for r in idx["patches"])
    print(f"[index] {len(idx['patches'])} recortes com trigo · {n5} com ≥{C.MIN_WHEAT_FRAC:.0%} de trigo")
    if a.check:
        old = json.loads(OUT.read_text())
        same = old["patches"] == idx["patches"]
        print("[index] idêntico ao commitado" if same else "[index] DIFERENTE do commitado")
        raise SystemExit(0 if same else 1)
    OUT.write_text(json.dumps(idx))
    print(f"[index] → {OUT}")
