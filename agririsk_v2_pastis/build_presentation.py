"""
Gera docs/AgriRisk_MA_v2_apresentacao.html a partir dos artefatos, para que todo número
da apresentação venha de um JSON reproduzível.

    python build_presentation.py
"""
import base64
import collections
import html
import json
from pathlib import Path

import cv2

import config as C

A = C.ARTIFACTS
m1 = json.loads((A / "agent01/metrics.json").read_text())
m2 = json.loads((A / "agent02/metrics.json").read_text())
var = json.loads((A / "agent02/audit_variants.json").read_text())
m3 = json.loads((A / "agent03/metrics.json").read_text())
llm = json.loads((A / "agent04/llm_eval.json").read_text())
samples = json.loads((A / "agent04/sample_pareceres.json").read_text())
bench = json.loads((A / "agent03/regional_benchmark.json").read_text())
reports = {p.stem: json.loads(p.read_text()) for p in (A / "reports").glob("*.json")}
payloads = [json.loads(p.read_text()) for p in (A / "payloads").glob("*.json")]

PARCEL_LOW = "40025_7275742"          # trigo real, vigor muito abaixo da região
PARCEL_INFL = "10270_1110439"         # trigo com área declarada +30%
PARCEL_MIS = "30455_13881546"         # milho declarado como trigo
EV_INFL, EV_OK = "10270_1110439_infl", "10259_1147481"

ORANGE, BLUE = "#de6c00", "#4f8fe6"   # validados (dataviz validator, modo escuro, superfície #0f1f38)
MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=0):
    return f"{100 * x:.{d}f}%".replace(".", ",")


def mlabel(iso):
    y, m, _ = iso.split("-")
    return f"{MONTHS[int(m) - 1]}/{y[2:]}"


def dlabel(iso):
    y, m, d = iso.split("-")
    return f"{int(d)}/{MONTHS[int(m) - 1]}"


def img_b64(name, max_w=560):
    im = cv2.imread(str(A / "agent02/evidence" / f"{name}.png"))
    if im.shape[1] > max_w:
        im = cv2.resize(im, (max_w, int(im.shape[0] * max_w / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def row(*cells, head=False):
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


STYLE = ('<style>.g{{stroke:#1c355e;stroke-width:1}}.ax{{fill:#8fa1be;font-size:11px;font-family:Inter,system-ui,sans-serif}}'
         '.lb{{fill:#c8d6e8;font-size:12px;font-family:Inter,system-ui,sans-serif}}</style>')


# ---------------------------------------------------------------------------
# Gráfico 1 · confirmação antecipada: % de trigos confirmados e precisão por data
# ---------------------------------------------------------------------------
def early_chart():
    e = m1["early_season"]["by_cutoff"]
    W, H, L, R, T, B = 760, 280, 46, 150, 16, 34
    n = len(e)
    x = lambda i: L + i * (W - L - R) / (n - 1)
    y = lambda v: T + (1 - v) * (H - T - B)
    grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{int(v * 100)}%</text>'
                   for v in (0, .25, .5, .75, 1))
    ticks = "".join(f'<text x="{x(i):.1f}" y="{H - 12}" class="ax" text-anchor="middle">{dlabel(r["date"])}</text>'
                    for i, r in enumerate(e))
    series = []
    for key, color, label in (("confirmed_share_of_wheat", ORANGE, "trigos já confirmados"),
                              ("precision_of_confirmed", BLUE, "precisão dos confirmados")):
        pts = " ".join(f"{x(i):.1f},{y(r[key]):.1f}" for i, r in enumerate(e))
        dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(r[key]):.1f}" r="4" fill="{color}"/>'
                       f'<circle cx="{x(i):.1f}" cy="{y(r[key]):.1f}" r="10" fill="transparent"><title>{r["date"]}: '
                       f'{label} {pct(r[key])}</title></circle>' for i, r in enumerate(e))
        series.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>{dots}'
                      f'<text x="{x(n - 1) + 10:.1f}" y="{y(e[-1][key]) + 4:.1f}" class="lb">{label}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Percentual de trigos confirmados e precisão por data de corte da safra">'
            f'{STYLE.replace("{{", "{").replace("}}", "}")}{grid}{ticks}{"".join(series)}</svg>')


# ---------------------------------------------------------------------------
# Gráfico 2 · NDVI da parcela vs faixa regional
# ---------------------------------------------------------------------------
def ndvi_chart(rep):
    a1, a3 = rep["agent01_output"], rep["agent03_output"]
    reg = bench["regions"][a3["region_benchmark_used"]]
    dates, vals = a1["ndvi_curve"]["dates"], a1["ndvi_curve"]["values"]
    P = reg["ndvi_percentiles"]
    W, H, L, R, T, B = 760, 300, 46, 130, 16, 34
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
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="NDVI da parcela comparado à faixa das parcelas de trigo da mesma região">'
            f'{STYLE.replace("{{", "{").replace("}}", "}")}{grid}{ticks}'
            f'<polygon points="{band("p10", "p90")}" fill="{BLUE}" opacity=".16"/>'
            f'<polygon points="{band("p25", "p75")}" fill="{BLUE}" opacity=".28"/>'
            f'<polyline points="{line(P["p50"])}" fill="none" stroke="{BLUE}" stroke-width="2"/>'
            f'<polyline points="{line(vals)}" fill="none" stroke="{ORANGE}" stroke-width="2.5"/>'
            f'<text x="{x(k) + 8:.1f}" y="{y(P["p50"][k]) - 6:.1f}" class="lb">mediana da região</text>'
            f'<text x="{x(k) + 8:.1f}" y="{y(vals[k]) + 14:.1f}" class="lb">esta parcela</text>{pts}</svg>')


# ---------------------------------------------------------------------------
# Gráfico 3 · atenção temporal média nos pixels de trigo
# ---------------------------------------------------------------------------
def attention_chart():
    d = m1["attention_on_wheat"]["dates"]
    heads = m1["attention_on_wheat"]["per_head"]
    a = [sum(h[i] for h in heads) / len(heads) for i in range(len(d))]
    W, H, L, B, T = 760, 200, 46, 34, 14
    n, mx = len(d), max(a)
    bw = (W - L - 10) / n
    bars = "".join(f'<rect x="{L + i * bw + 1:.1f}" y="{T + (1 - v / mx) * (H - T - B):.1f}" width="{bw - 2:.1f}" '
                   f'height="{max((v / mx) * (H - T - B), 1):.1f}" rx="3" fill="{ORANGE}"><title>{d[i]}: {pct(v, 1)} da atenção</title></rect>'
                   for i, v in enumerate(a))
    ticks = "".join(f'<text x="{L + i * bw + bw / 2:.1f}" y="{H - 12}" class="ax" text-anchor="middle">{mlabel(x)}</text>'
                    for i, x in enumerate(d) if i % 3 == 0)
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peso médio da atenção temporal por data nos pixels de trigo">'
            f'{STYLE.replace("{{", "{").replace("}}", "}")}<line x1="{L}" x2="{W - 10}" y1="{H - B}" y2="{H - B}" stroke="#1c355e"/>'
            f'{bars}{ticks}</svg>')


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
early = m1["early_season"]["by_cutoff"]
early_rows = "".join(row(dlabel(e["date"]) + "/" + e["date"][2:4], pct(e["confirmed_share_of_wheat"]),
                         pct(e["precision_of_confirmed"]), f(e["wheat_f1"], 3), f(e["wheat_auc"], 3))
                     for e in early if e["cutoff_step"] in (9, 13, 15, 17, 21, 25))
apr = next(e for e in early if e["date"].startswith("2019-04"))
may = next(e for e in early if e["date"].startswith("2019-05"))

vr = var["results"]
var_rows = "".join(row(f"<b>{k}</b>" + (" ✓" if k == var["chosen_on_validation"] else ""), html.escape(var["description"][k]),
                       pct(vr[k]["val_det_30"], 1), f(vr[k]["val_auc_30"], 3), pct(vr[k]["test_det_30"], 1),
                       pct(vr[k]["test_det_15"], 1), pct(vr[k]["test_false_alarm"], 1)) for k in "ABC")
aa = m2["area_audit"]
inst = m2.get("test", {})

sc = collections.defaultdict(collections.Counter)
for r in payloads:
    a3 = r["agent03_output"]
    s = r["scenario"]
    sc[s]["n"] += 1
    sc[s][a3["risk_level"]] += 1
    sc[s]["decl_flag"] += not a3["declaration_check"]["consistent"]
    sc[s]["area_flag"] += a3["area_audit"]["declared_area_supported"] is False
    sc[s]["area_na"] += a3["area_audit"]["declared_area_supported"] is None
names = {"honesto": "Declaração honesta", "cultura_falsa": "Outra cultura declarada como trigo",
         "area_inflada": "Trigo com área declarada +30%"}
scen_rows = "".join(row(names[s], str(c["n"]), str(c["decl_flag"]), f'{c["area_flag"]} <span class="small">({c["area_na"]} não auditáveis)</span>',
                        f'{c["high"]} / {c["medium"]} / {c["low"]}') for s, c in
                    sorted(sc.items(), key=lambda kv: ["honesto", "cultura_falsa", "area_inflada"].index(kv[0])))

llm_rows = "".join(row(f"<b>{k}</b>" if k == C.LLM_MODEL else k, pct(v["validation_pass_rate"]), pct(v["risk_level_agreement"]),
                       f"{f(v['latency_s_median'], 1)} s", f"{v['prompt_tokens_mean']} + {v['completion_tokens_mean']}")
                   for k, v in llm.items())
n_llm = next(iter(llm.values()))["n"]
samp = {(s["model"], s["scenario"]): s for s in samples}


def sample_text(model, scen="area_inflada"):
    s = samp.get((model, scen))
    return html.escape(s["diagnostico"]) if s else "—"


dc = m3["declaration_check"]
pix1, par1, rf1 = m1["test_pixel"], m1["test_parcel"], m1["baseline_rf_parcel"]
rep_low, rep_inf, rep_mis = reports.get(PARCEL_LOW), reports[PARCEL_INFL], reports[PARCEL_MIS]
if rep_low is None:                          # o exemplo de vigor vem das evidências, se não houver relatório
    rep_low = next(p for p in payloads if f'{p["patch"]}_{p["parcel"]}' == PARCEL_LOW)
rc = rep_low["agent03_output"]["regional_comparison"]
a3i, a4i = rep_inf["agent03_output"], rep_inf["agent04_output"]
aai = a3i["area_audit"]
recs_inf = "".join(f'<li><b>[{r["prioridade"]}]</b> {html.escape(r["acao"])}<br><span class="accent small">↳ {r["evidencia_referenciada"]}</span></li>'
                   for r in a4i["recomendacoes"])
a3m, a4m = rep_mis["agent03_output"], rep_mis["agent04_output"]

subs = {
    "EARLY_CHART": early_chart(), "NDVI_CHART": ndvi_chart(rep_low), "ATT_CHART": attention_chart(),
    "IMG_INFL": img_b64(EV_INFL), "IMG_OK": img_b64(EV_OK),
    "A1_PARAMS": f"{m1['params'] / 1e6:.2f}".replace(".", ","), "A1_MIN": f"{m1['train_seconds'] / 60:.0f}",
    "A1_PIX_IOU": f(pix1["wheat_iou"]), "A1_PIX_F1": f(pix1["wheat_f1"]), "A1_MIOU": f(pix1["miou"]),
    "A1_PAR_F1": f(par1["wheat_f1"], 3), "A1_PAR_AUC": f(par1["wheat_auc"], 3), "A1_PAR_P": f(par1["wheat_precision"], 3),
    "A1_PAR_R": f(par1["wheat_recall"], 3),
    "RF_PAR_F1": f(rf1["wheat_f1"], 3), "RF_PAR_AUC": f(rf1["wheat_auc"], 3), "RF_PAR_P": f(rf1["wheat_precision"], 3),
    "RF_PAR_R": f(rf1["wheat_recall"], 3),
    "N_TEST_PARCELS": str(par1["n_parcels"]), "N_TEST_WHEAT": str(par1["n_wheat"]),
    "EARLY_ROWS": early_rows, "APR_SHARE": pct(apr["confirmed_share_of_wheat"]), "APR_PREC": pct(apr["precision_of_confirmed"]),
    "APR_DATE": dlabel(apr["date"]), "MAY_SHARE": pct(may["confirmed_share_of_wheat"]), "MAY_PREC": pct(may["precision_of_confirmed"]),
    "A2_EXT": f(m2["test"]["extent_iou"]), "A2_BF1": f(m2["test"]["boundary_f1"]),
    "A2_BP": f(m2["test"]["boundary_precision"]), "A2_BR": f(m2["test"]["boundary_recall"]),
    "A2_PIOU": f(inst["parcel_iou_median"]), "A2_REC": pct(inst["parcels_recovered_iou50"]),
    "A2_MIN": f"{m2['train_seconds'] / 60:.1f}".replace(".", ","),
    "VAR_ROWS": var_rows, "VAR_CHOSEN": var["chosen_on_validation"],
    "AA_DET30": pct(aa["detection_rate_inflated_30pct"], 1), "AA_DET15": pct(aa["detection_rate_inflated_15pct"], 1),
    "AA_FA": pct(aa["honest_false_alarm_rate_test"], 1), "AA_AUC30": f(aa["auc_inflated_30pct"]), "AA_N": str(aa["n_test_parcels"]),
    "AA_DETA": pct(vr["A"]["test_det_30"], 1),
    "DC_RECALL": pct(dc["test_recall_of_mismatch"]), "DC_FA": f(dc["alerts_per_100_true_wheat_parcels"], 1),
    "DC_PREC5": pct(dc["expected_precision_at_5pct_prevalence"]), "DC_TAU": f(dc["tau"]),
    "SCEN_ROWS": scen_rows, "N_PAYLOADS": str(len(payloads)),
    "BENCH_N": str(m3["benchmark_parcels_per_region"]["ALL"]),
    "LOW_PCT": f(rc["ndvi_integral_percentile"], 1), "LOW_PEAK": f(rc["peak_ndvi"]), "LOW_PEAK_PCT": f(rc["peak_ndvi_percentile"], 1),
    "LOW_DELAY": str(rc["peak_delay_days_vs_region"]), "LOW_N_REG": str(rc["n_wheat_parcels_in_region_train"]),
    "LLM_ROWS": llm_rows, "LLM_N": str(n_llm), "LLM_MODEL": C.LLM_MODEL,
    "S_4O": sample_text("gpt-4o-mini"), "S_41": sample_text("gpt-4.1-mini"), "S_54": sample_text("gpt-5.4-mini"),
    "INF_DECL": f(aai["declared_area_ha"]), "INF_REG": f(aai["registry_area_ha"]), "INF_SCORE": f(aai["inconsistency_score"], 1),
    "INF_TAU": f(aai["threshold_inconsistency"], 1), "INF_P": f(a3i["declaration_check"]["p_declared"], 3),
    "INF_CONF": dlabel(a3i["declaration_check"]["confirmed_since"]) if a3i["declaration_check"]["confirmed_since"] else "—",
    "INF_SCORE_RISK": f(a3i["risk_score"]), "INF_LEVEL": a4i["nivel_risco"],
    "INF_DIAG": html.escape(a4i["diagnostico"]), "INF_RECS": recs_inf,
    "MIS_P": f(a3m["declaration_check"]["p_declared"], 3), "MIS_ALT": a3m["declaration_check"]["temporal_model_alternative"],
    "MIS_DIAG": html.escape(a4m["diagnostico"]),
    # leitura humana dos três pareceres acima (artifacts/agent04/sample_pareceres.json)
    "LLM_QUAL": ("<b>Análise de erros:</b> o gpt-4o-mini atribui à imagem a área de 1,91 ha, que é a do cadastro "
                 "(a imagem não mede área), e lê 1,21% de manchas como “vigor baixo”. O gpt-4.1-mini repete a "
                 "confusão imagem × cadastro e trata a pontuação de inconsistência como porcentagem; suas falhas "
                 "automáticas vêm de contas que ele fez (somas e diferenças fora do payload). O gpt-5.4-mini separa "
                 "declarado × cadastro e cita o percentil regional certo, com um erro de digitação. Custo da escolha: "
                 "~1,7× mais tokens de saída, com latência igual."),
}
out = Path(__file__).with_name("presentation_template.html").read_text()
for k, v in subs.items():
    out = out.replace("{{" + k + "}}", v)
assert "{{" not in out, [l for l in out.splitlines() if "{{" in l][:3]
dst = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(out, encoding="utf-8")
print(f"ok → {dst} ({dst.stat().st_size / 1024:.0f} KB)")
