"""
Gera docs/AgriRisk_MA_v2_apresentacao.html a partir dos artefatos de treino/avaliação,
para que todo número da apresentação venha de um JSON reproduzível.

    python build_presentation.py
"""
import base64
import html
import json
from pathlib import Path

import cv2

import config as C

A = C.ARTIFACTS
m1 = json.loads((A / "agent01/metrics.json").read_text())
m2 = json.loads((A / "agent02/metrics.json").read_text())
m3 = json.loads((A / "agent03/metrics.json").read_text())
llm = json.loads((A / "agent04/llm_eval.json").read_text())
bench = json.loads((A / "agent03/regional_benchmark.json").read_text())
reports = {p.stem: json.loads(p.read_text()) for p in (A / "reports").glob("*.json")}

PARCEL_LOW = "40067_7349456"      # trigo real com vigor muito abaixo da região
PARCEL_OK = "10084_14157845"      # trigo saudável (comparação visual)
PARCEL_MIS = "40005_7278188"      # milho declarado como trigo

ORANGE, BLUE = "#de6c00", "#4f8fe6"     # validados (dataviz validator, modo escuro, superfície #0f1f38)


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=0):
    return f"{100 * x:.{d}f}%".replace(".", ",")


def img_b64(path, max_w=520):
    im = cv2.imread(str(path))
    if im.shape[1] > max_w:
        im = cv2.resize(im, (max_w, int(im.shape[0] * max_w / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def mlabel(iso):
    y, m, _ = iso.split("-")
    return f"{MONTHS[int(m) - 1]}/{y[2:]}"


# ---------------------------------------------------------------------------
# Gráfico 1 · curva de NDVI da parcela vs faixa regional
# ---------------------------------------------------------------------------
def ndvi_chart(rep):
    a1, a3 = rep["agent01_output"], rep["agent03_output"]
    reg = bench["regions"][a3["region_benchmark_used"]]
    dates, vals = a1["ndvi_curve"]["dates"], a1["ndvi_curve"]["values"]
    P = reg["ndvi_percentiles"]
    W, H, L, R, T, B = 760, 300, 46, 120, 16, 34
    n = len(dates)
    x = lambda i: L + i * (W - L - R) / (n - 1)
    y = lambda v: T + (1 - v) * (H - T - B)
    band = lambda lo, hi: " ".join(f"{x(i):.1f},{y(P[hi][i]):.1f}" for i in range(n)) + " " + \
        " ".join(f"{x(i):.1f},{y(P[lo][i]):.1f}" for i in reversed(range(n)))
    line = lambda vs: " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vs))
    grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{f(v, 1)}</text>' for v in (0, .2, .4, .6, .8, 1))
    ticks = "".join(f'<text x="{x(i):.1f}" y="{H - 12}" class="ax" text-anchor="middle">{mlabel(d)}</text>'
                    for i, d in enumerate(dates) if i % 3 == 0)
    pts = "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="9" fill="transparent"><title>{d}: parcela {f(v)} · '
                  f'mediana da região {f(P["p50"][i])}</title></circle>' for i, (d, v) in enumerate(zip(dates, vals)))
    k = n - 1
    return f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="NDVI da parcela comparado à faixa das parcelas de trigo da mesma região">
  <style>.g{{stroke:#1c355e;stroke-width:1}}.ax{{fill:#8fa1be;font-size:11px;font-family:Inter,system-ui,sans-serif}}
  .lb{{fill:#c8d6e8;font-size:12px;font-family:Inter,system-ui,sans-serif}}</style>
  {grid}{ticks}
  <polygon points="{band("p10", "p90")}" fill="{BLUE}" opacity=".16"/>
  <polygon points="{band("p25", "p75")}" fill="{BLUE}" opacity=".28"/>
  <polyline points="{line(P["p50"])}" fill="none" stroke="{BLUE}" stroke-width="2"/>
  <polyline points="{line(vals)}" fill="none" stroke="{ORANGE}" stroke-width="2.5"/>
  <text x="{x(k) + 8:.1f}" y="{y(P["p50"][k]) - 6:.1f}" class="lb">mediana da região</text>
  <text x="{x(k) + 8:.1f}" y="{y(vals[k]) + 14:.1f}" class="lb">esta parcela</text>
  {pts}
</svg>'''


# ---------------------------------------------------------------------------
# Gráfico 2 · atenção temporal média do U-TAE-lite nos pixels de trigo
# ---------------------------------------------------------------------------
def attention_chart():
    d = m1["attention_on_wheat"]["dates"]
    a = [sum(h[i] for h in m1["attention_on_wheat"]["per_head"]) / len(m1["attention_on_wheat"]["per_head"])
         for i in range(len(d))]
    W, H, L, B, T = 760, 220, 46, 34, 14
    n, mx = len(d), max(a)
    bw = (W - L - 10) / n
    bars = "".join(
        f'<rect x="{L + i * bw + 1:.1f}" y="{T + (1 - v / mx) * (H - T - B):.1f}" width="{bw - 2:.1f}" '
        f'height="{max((v / mx) * (H - T - B), 1):.1f}" rx="3" fill="{ORANGE}"><title>{d[i]}: {pct(v, 1)} da atenção</title></rect>'
        for i, v in enumerate(a))
    ticks = "".join(f'<text x="{L + i * bw + bw / 2:.1f}" y="{H - 12}" class="ax" text-anchor="middle">{mlabel(x)}</text>'
                    for i, x in enumerate(d) if i % 3 == 0)
    return f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peso médio da atenção temporal por data nos pixels de trigo">
  <style>.ax{{fill:#8fa1be;font-size:11px;font-family:Inter,system-ui,sans-serif}}</style>
  <line x1="{L}" x2="{W - 10}" y1="{H - B}" y2="{H - B}" stroke="#1c355e"/>{bars}{ticks}
</svg>'''


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def row(*cells, head=False):
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


wd = m3["wheat_detection_test"]
fusion_rows = "".join(row(name, f(v["auc"], 3), f(v["f1"], 3), f(v["precision"], 3), f(v["recall"], 3)) for name, v in
                      [("Só série temporal (Agent 01)", wd["temporal_only"]), ("Só imagem SPOT (Agent 02)", wd["visual_only"]),
                       ("<b>Fusão aprendida (Agent 03)</b>", wd["fused"])])
dc = m3["declaration_check"]
llm_rows = "".join(row(f"<b>{k}</b>" if k == C.LLM_MODEL else k, pct(v["validation_pass_rate"]), pct(v["risk_level_agreement"]),
                       f"{f(v['latency_s_median'], 1)} s", f"{v['prompt_tokens_mean']} + {v['completion_tokens_mean']}")
                   for k, v in llm.items())
n_llm = next(iter(llm.values()))["n"]
rep_low, rep_mis = reports[PARCEL_LOW], reports[PARCEL_MIS]
a3l, a4l = rep_low["agent03_output"], rep_low["agent04_output"]
rc = a3l["regional_comparison"]
recs_low = "".join(f'<li><b>[{r["prioridade"]}]</b> {html.escape(r["acao"])}<br><span class="accent small">↳ {r["evidencia_referenciada"]}</span></li>'
                   for r in a4l["recomendacoes"])
a3m, a4m = rep_mis["agent03_output"], rep_mis["agent04_output"]
pix1, par1, rf1 = m1["test_pixel"], m1["test_parcel"], m1["baseline_rf_parcel"]
sample = json.loads((A / "agent04/sample_mismatch_pareceres.json").read_text())

TEMPLATE = Path(__file__).with_name("presentation_template.html").read_text()
out = TEMPLATE
subs = {
    "NDVI_CHART": ndvi_chart(rep_low), "ATT_CHART": attention_chart(),
    "IMG_LOW": img_b64(A / "agent02/evidence" / f"{PARCEL_LOW}.png"),
    "IMG_OK": img_b64(A / "agent02/evidence" / f"{PARCEL_OK}.png"),
    "A1_PARAMS": f"{m1['params'] / 1e6:.2f}".replace(".", ","), "A1_MIN": f"{m1['train_seconds'] / 60:.0f}",
    "A1_PIX_IOU": f(pix1["wheat_iou"]), "A1_PIX_F1": f(pix1["wheat_f1"]), "A1_MIOU": f(pix1["miou"]), "A1_OA": f(pix1["overall_accuracy"]),
    "A1_PAR_F1": f(par1["wheat_f1"], 3), "A1_PAR_AUC": f(par1["wheat_auc"], 3), "A1_PAR_P": f(par1["wheat_precision"], 3), "A1_PAR_R": f(par1["wheat_recall"], 3),
    "RF_PAR_F1": f(rf1["wheat_f1"], 3), "RF_PAR_AUC": f(rf1["wheat_auc"], 3), "RF_PAR_P": f(rf1["wheat_precision"], 3), "RF_PAR_R": f(rf1["wheat_recall"], 3),
    "N_TEST_PARCELS": str(par1["n_parcels"]), "N_TEST_WHEAT": str(par1["n_wheat"]),
    "A2_PIX_IOU": f(m2["test_pixel"]["wheat_iou"]), "A2_PIX_F1": f(m2["test_pixel"]["wheat_f1"]),
    "A2_PAR_F1": f(m2["test_parcel"]["wheat_f1"], 3), "A2_PAR_AUC": f(m2["test_parcel"]["wheat_auc"], 3),
    "A2_MIN": f"{m2['train_seconds'] / 60:.1f}".replace(".", ","), "A2_SPOT_MED": f(m2["low_vigor_pct_wheat_test"]["median"], 1),
    "A2_SPOT_P90": f(m2["low_vigor_pct_wheat_test"]["p90"], 1),
    "FUSION_ROWS": fusion_rows, "COEF_T": f(m3["fusion_coefficients"]["temporal"]), "COEF_V": f(m3["fusion_coefficients"]["visual"]),
    "DC_RECALL": pct(dc["test_recall_of_mismatch"]), "DC_FA": f(dc["alerts_per_100_true_wheat_parcels"], 1),
    "DC_PREC5": pct(dc["expected_precision_at_5pct_prevalence"]), "DC_TAU": f(dc["tau"]),
    "BENCH_N": str(m3["benchmark_parcels_per_region"]["ALL"]),
    "LLM_ROWS": llm_rows, "LLM_N": str(n_llm), "LLM_MODEL": C.LLM_MODEL,
    "S_4O": html.escape(sample["gpt-4o-mini"]["diagnostico"]), "S_41": html.escape(sample["gpt-4.1-mini"]["diagnostico"]),
    "S_54": html.escape(sample["gpt-5.4-mini"]["diagnostico"]),
    "LOW_P": f(a3l["declaration_check"]["p_declared_fused"], 3), "LOW_PCT": f(rc["ndvi_integral_percentile"], 1),
    "LOW_PEAK": f(rc["peak_ndvi"]), "LOW_PEAK_PCT": f(rc["peak_ndvi_percentile"], 1), "LOW_DELAY": str(rc["peak_delay_days_vs_region"]),
    "LOW_N_REG": str(rc["n_wheat_parcels_in_region_train"]), "LOW_SCORE": f(a3l["risk_score"]), "LOW_LEVEL": a4l["nivel_risco"],
    "LOW_AREA": f(a3l["area_ha"], 1), "LOW_DIAG": html.escape(a4l["diagnostico"]), "LOW_RECS": recs_low,
    "MIS_P": f(a3m["declaration_check"]["p_declared_fused"], 3), "MIS_ALT": a3m["declaration_check"]["temporal_model_alternative"],
    "MIS_DIAG": html.escape(a4m["diagnostico"]),
    "OK_PCT": f(reports[PARCEL_OK]["agent03_output"]["regional_comparison"]["ndvi_integral_percentile"], 1),
}
for k, v in subs.items():
    out = out.replace("{{" + k + "}}", v)
assert "{{" not in out, [l for l in out.splitlines() if "{{" in l][:3]
dst = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(out, encoding="utf-8")
print(f"ok → {dst} ({dst.stat().st_size / 1024:.0f} KB)")
