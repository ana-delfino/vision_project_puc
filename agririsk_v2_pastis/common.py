"""Utilidades compartilhadas pelos agentes da v2."""
import json
from datetime import timedelta
from functools import lru_cache

import numpy as np

import config as C

LABELS_EN = {int(k): v for k, v in json.loads((C.INDEX / "label_names.json").read_text()).items()}
LABELS_PT = {
    0: "fundo (não agrícola)", 1: "pastagem", 2: "trigo de inverno", 3: "milho", 4: "cevada de inverno",
    5: "canola de inverno", 6: "cevada de primavera", 7: "girassol", 8: "videira", 9: "beterraba",
    10: "triticale", 11: "trigo duro", 12: "frutas/hortaliças/flores", 13: "batata",
    14: "forrageira leguminosa", 15: "soja", 16: "pomar", 17: "cereal misto", 18: "sorgo", 19: "void",
}
GRID_DATES = [C.T_START + timedelta(days=k * C.T_STEP_DAYS) for k in range(C.T_STEPS)]


def manifest(split=None):
    rows = json.loads((C.DATA / "manifest.json").read_text())
    return [r for r in rows if split is None or r["fold"] in C.FOLDS[split]]


@lru_cache(maxsize=64)
def load_patch(patch: int):
    d = np.load(C.DATA / "patches" / f"{patch}.npz")
    return {k: d[k] for k in d.files}


def ndvi(red, nir):
    red, nir = np.asarray(red, np.float32), np.asarray(nir, np.float32)
    return (nir - red) / (nir + red + 1e-6)


def parcel_ndvi(curves: np.ndarray) -> np.ndarray:
    """curves (..., T, 10) → NDVI (..., T)."""
    return ndvi(curves[..., C.B_RED], curves[..., C.B_NIR])


def smooth(v: np.ndarray) -> np.ndarray:
    """Mediana móvel de 3 passos: remove quedas de uma única data (nuvem/sombra
    residual) sem apagar a colheita, que é uma queda sustentada."""
    p = np.pad(np.asarray(v, np.float32), 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def phenology(ndvi_curve: np.ndarray) -> dict:
    """Métricas fenológicas do trigo a partir da curva de NDVI da grade de 16 dias (suavizada).

    O pico é procurado só na janela em que o pico do trigo pode ocorrer (jan-jul/2019), e os
    limites de meia-amplitude (início e fim do ciclo verde) só dentro do ciclo da safra. Sem isso,
    uma cultura de cobertura ou rebrota após a colheita (verde em set-out) podia ser tomada como
    o pico do trigo, gerando "atrasos" de meses e vigor artificialmente baixo. Se a suavização
    deixar datas empatadas no máximo, o pico é o centro do platô (desempate pelo NDVI bruto)."""
    v = smooth(ndvi_curve)
    steps = np.arange(len(v))
    in_season = np.array([C.CROP_SEASON[0] <= d <= C.CROP_SEASON[1] for d in GRID_DATES])
    in_peak = np.array([C.CROP_PEAK_WINDOW[0] <= d <= C.CROP_PEAK_WINDOW[1] for d in GRID_DATES])
    # empate na curva suavizada (platô da mediana móvel): centro do platô; com número par de
    # datas empatadas, das duas centrais fica a de maior NDVI bruto
    raw = np.asarray(ndvi_curve, np.float32)
    cand = steps[in_peak][np.isclose(v[in_peak], v[in_peak].max(), atol=1e-6)]
    mid = cand[(len(cand) - 1) // 2: len(cand) // 2 + 1]
    k = int(mid[raw[mid].argmax()])
    s0, s1 = int(steps[in_season][0]), int(steps[in_season][-1])
    lo = float(np.percentile(v[in_season], 10))
    half = lo + 0.5 * (v[k] - lo)
    up = next((i for i in range(k, s0 - 1, -1) if v[i] < half), s0)
    down = next((i for i in range(k, s1 + 1) if v[i] < half), s1)
    return {
        "peak_ndvi": round(float(v[k]), 3),
        "peak_date": GRID_DATES[k].isoformat(), "peak_step": k,
        "greenup_date": GRID_DATES[up].isoformat(),
        "senescence_date": GRID_DATES[down].isoformat(),
        "green_season_days": int((down - up) * C.T_STEP_DAYS),
        "ndvi_integral": round(float(v[up:down + 1].sum() * C.T_STEP_DAYS / 30), 2),  # NDVI·mês
    }


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
