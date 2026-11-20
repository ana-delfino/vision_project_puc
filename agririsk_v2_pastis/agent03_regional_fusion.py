"""
Agent 03 — Regional Context & Risk Agent
AgriRisk-MA · Junta "o que está plantado" (Agent 01), "quanto está plantado" (Agent 02)
             e "como está comparado aos vizinhos" (benchmark regional) num risco auditável

1. Conferência da cultura declarada
   P(trigo) do Agent 01 (safra completa) < τ → declaração não confirmada. τ escolhido na
   VALIDAÇÃO como o maior limiar com ≤ 3% de alarmes em trigos reais (custo de vistoria);
   precisão também sob prevalência de 5% (premissa).
   Inclui a 1ª data em que a cultura já estava confirmada (confirmação antecipada).

2. Auditoria de área (Agent 02)
   área delimitada ÷ área declarada < τ_área → área declarada não sustentada pela imagem.

3. Benchmark regional (folds de TREINO, por região/tile Sentinel-2)
   Distribuições das parcelas de trigo: curva de NDVI (p10…p90), integral de NDVI, NDVI
   de pico, data do pico e % em manchas de baixo vigor (SPOT, dentro do polígono declarado).
   Cada parcela recebe seu PERCENTIL dentro da própria região.

4. Score de risco auditável (0-1) com decomposição por componente.

Uso
  python agent03_regional_fusion.py      # benchmark + limiar de declaração (após treinar 01 e 02)
"""
import json
import math

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config as C
from common import GRID_DATES, load_patch, manifest, parcel_ndvi, phenology

OUT = C.ARTIFACTS / "agent03"
MIN_REGION_PARCELS = 30
PREVALENCE = 0.05            # premissa de cenário: 5% das declarações de trigo não são trigo
WEIGHTS = {"declaration": 0.35, "area": 0.20, "vigor": 0.25, "timing": 0.10, "heterogeneity": 0.10}


# ---------------------------------------------------------------------------
# 1. BENCHMARK REGIONAL (só folds de treino)
# ---------------------------------------------------------------------------
def build_benchmark():
    from agent02_field_delineation import declared_mask, low_vigor_spots
    per_region = {}
    for r in manifest("train"):
        d = load_patch(r["patch"])
        for pid, lab, cur in zip(d["parcel_ids"], d["parcel_labels"], d["parcel_curves"]):
            if lab != C.CROP_ID:
                continue
            ph = phenology(parcel_ndvi(cur.astype(np.float32)))
            m = declared_mask(r["patch"], int(pid))
            s, _ = low_vigor_spots(d["spot"], m) if m.sum() >= 50 else ({"low_vigor_pct": None}, None)
            e = per_region.setdefault(r["tile"], {"curves": [], "peak_ndvi": [], "ndvi_integral": [],
                                                  "peak_step": [], "low_vigor_pct": []})
            e["curves"].append(parcel_ndvi(cur.astype(np.float32)))
            for k in ("peak_ndvi", "ndvi_integral", "peak_step"):
                e[k].append(ph[k])
            if s["low_vigor_pct"] is not None:
                e["low_vigor_pct"].append(s["low_vigor_pct"])
    bench = {"dates": [g.isoformat() for g in GRID_DATES], "regions": {}}
    allr = {k: [] for k in ("curves", "peak_ndvi", "ndvi_integral", "peak_step", "low_vigor_pct")}
    for e in per_region.values():
        for k in allr:
            allr[k] += e[k]
    for name, e in list(per_region.items()) + [("ALL", allr)]:
        cv = np.array(e["curves"])
        bench["regions"][name] = {
            "n_parcels": len(cv),
            "ndvi_percentiles": {f"p{q}": np.round(np.percentile(cv, q, axis=0), 3).tolist() for q in (10, 25, 50, 75, 90)},
            **{k: sorted(np.round(e[k], 3).tolist()) for k in ("peak_ndvi", "ndvi_integral", "peak_step", "low_vigor_pct")},
        }
    return bench


def percentile_of(value, dist):
    if value is None or not dist:
        return None
    return round(100.0 * np.searchsorted(dist, value, side="right") / len(dist), 1)


# ---------------------------------------------------------------------------
# 2. LIMIAR DA CONFERÊNCIA DE DECLARAÇÃO (validação) e AVALIAÇÃO (teste)
# ---------------------------------------------------------------------------
def _preds(split):
    t = json.loads((C.ARTIFACTS / "agent01" / f"parcel_preds_{split}.json").read_text())
    return np.array([x["p_wheat"] for x in t]), np.array([x["label"] == C.CROP_ID for x in t])


MAX_FALSE_ALARM = 0.03       # restrição de negócio: no máximo 3% dos trigos reais acusados (validação)


def fit_declaration():
    """τ = maior limiar com alarme falso ≤ 3% nos trigos reais da validação (maximiza a
    detecção de declarações falsas respeitando o custo de vistorias desnecessárias)."""
    pv, yv = _preds("val")
    grid = np.linspace(0.01, 0.99, 99)
    ok = [t for t in grid if (pv[yv] < t).mean() <= MAX_FALSE_ALARM]
    tau = float(max(ok)) if ok else float(grid[0])
    pt, yt = _preds("test")
    mis_pred, mis_true = pt < tau, ~yt
    tpr = float(recall_score(mis_true, mis_pred))
    fpr = float((mis_pred & yt).sum() / max(yt.sum(), 1))
    return {"tau": tau}, {
        "n_test_parcels": int(len(yt)), "n_test_wheat": int(yt.sum()),
        "wheat_detection_test": {"auc": round(float(roc_auc_score(yt, pt)), 4),
                                 "f1": round(float(f1_score(yt, pt > 0.5)), 4),
                                 "precision": round(float(precision_score(yt, pt > 0.5)), 4),
                                 "recall": round(float(recall_score(yt, pt > 0.5)), 4)},
        "declaration_check": {
            "tau": round(tau, 4), "test_recall_of_mismatch": round(tpr, 4),
            "test_false_alarm_rate_on_true_wheat": round(fpr, 4),
            f"expected_precision_at_{int(PREVALENCE * 100)}pct_prevalence":
                round(tpr * PREVALENCE / max(tpr * PREVALENCE + fpr * (1 - PREVALENCE), 1e-9), 4),
            "alerts_per_100_true_wheat_parcels": round(100 * fpr, 1)},
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = build_benchmark()
    (OUT / "regional_benchmark.json").write_text(json.dumps(bench))
    fit, metrics = fit_declaration()
    (OUT / "declaration.json").write_text(json.dumps(fit, indent=1))
    metrics["benchmark_parcels_per_region"] = {k: v["n_parcels"] for k, v in bench["regions"].items()}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 3. INFERÊNCIA: CONTEXTO REGIONAL + RISCO
# ---------------------------------------------------------------------------
_CACHE = {}


def _load():
    if not _CACHE:
        _CACHE["bench"] = json.loads((OUT / "regional_benchmark.json").read_text())
        _CACHE["tau"] = json.loads((OUT / "declaration.json").read_text())["tau"]
    return _CACHE


def level(score):
    return "high" if score >= 0.55 else "medium" if score >= 0.35 else "low"


def fuse_agent_outputs(a1: dict, a2: dict) -> dict:
    c = _load()
    p = a1["p_declared_crop"]
    mismatch = p < c["tau"]
    reg = c["bench"]["regions"].get(a1["region"])
    flags = []
    if reg is None or reg["n_parcels"] < MIN_REGION_PARCELS:
        flags.append("regional_benchmark_small_using_all_regions")
        reg, region_used = c["bench"]["regions"]["ALL"], "ALL"
    else:
        region_used = a1["region"]

    ph = a1["phenology"]
    pct_integral = percentile_of(ph["ndvi_integral"], reg["ndvi_integral"])
    pct_peak = percentile_of(ph["peak_ndvi"], reg["peak_ndvi"])
    median_step = float(np.median(reg["peak_step"]))
    delay = int(round((ph["peak_step"] - median_step) * C.T_STEP_DAYS))
    lv = a2["low_vigor"]["low_vigor_pct"]
    pct_spots = percentile_of(lv, reg["low_vigor_pct"])
    aa = a2["area_audit"]

    # --- componentes (0-1) ---------------------------------------------------
    comp = {
        "declaration": 1.0 if mismatch else round(max(0.0, (0.9 - p) / 0.9), 3),
        "area": 1.0 if not aa["declared_area_supported"] else round(min(1.0, max(0.0, 1 - aa["delineated_to_declared_ratio"]) / 0.3), 3),
        "vigor": (0.9 if pct_integral is not None and pct_integral < 10 else 0.6 if pct_integral is not None and pct_integral < 25
                  else 0.3 if pct_integral is not None and pct_integral < 50 else 0.1),
        "timing": round(min(1.0, abs(delay) / 48), 3),
        "heterogeneity": (0.8 if pct_spots is not None and pct_spots > 90 else 0.5 if pct_spots is not None and pct_spots > 75 else 0.1),
    }
    if mismatch:                       # outra cultura: comparar com trigos da região não faz sentido
        comp["vigor"], comp["timing"], comp["heterogeneity"] = 0.0, 0.0, 0.0
    score = sum(WEIGHTS[k] * v for k, v in comp.items())
    if mismatch:
        score = max(score, 0.70); flags.append("declared_crop_not_confirmed")
    if not aa["declared_area_supported"]:
        score = max(score, 0.55); flags.append("declared_area_not_supported_by_image")
    if aa["parcel_cut_by_image_edge"]:
        flags.append("parcel_cut_by_image_edge_area_unreliable")
    if lv is None:
        flags.append("parcel_too_small_for_vhr_analysis")
    flags.append("vigor_not_validated_against_yield")

    ec = a1.get("early_confirmation", {})
    return {
        "patch": a1["patch"], "parcel_id": a1["parcel_id"], "region": a1["region"], "region_benchmark_used": region_used,
        "declaration_check": {"declared_crop": a1["declared_crop"], "p_declared": p, "threshold_tau": round(c["tau"], 3),
                              "consistent": not mismatch, "temporal_model_alternative": a1["observed_crop"],
                              "confirmed_since": ec.get("confirmed_since")},
        "area_audit": aa,
        "regional_comparison": {
            "applicable": not mismatch,
            "n_wheat_parcels_in_region_train": reg["n_parcels"],
            "ndvi_integral": ph["ndvi_integral"], "ndvi_integral_percentile": pct_integral,
            "peak_ndvi": ph["peak_ndvi"], "peak_ndvi_percentile": pct_peak,
            "peak_date": ph["peak_date"], "region_median_peak_date": GRID_DATES[int(round(median_step))].isoformat(),
            "peak_delay_days_vs_region": delay,
            "low_vigor_pct": lv, "low_vigor_percentile": pct_spots,
        },
        "risk_score": round(float(score), 3), "risk_level": level(score),
        "score_breakdown": {**comp, "weights": WEIGHTS},
        "data_quality_flags": flags,
    }


if __name__ == "__main__":
    build()
