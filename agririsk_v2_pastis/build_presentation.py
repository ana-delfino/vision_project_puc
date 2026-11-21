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

PARCEL_LOW = "30455_50323"            # trigo real, vigor muito abaixo da região
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
    W, H, L, R, T, B = 760, 280, 46, 190, 16, 34
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
# Gráficos técnicos
# ---------------------------------------------------------------------------
CAT = ["#de6c00", "#4f8fe6", "#2f9e6e", "#a77bdc"]   # ordem fixa · validados (modo escuro, superfície #0f1f38)
CSS = STYLE.replace("{{", "{").replace("}}", "}")


def spread_labels(ys, min_gap=15):
    """Afasta rótulos diretos que colidiriam (mantém a ordem vertical)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out, last = list(ys), -1e9
    for i in order:
        out[i] = max(out[i], last + min_gap)
        last = out[i]
    return out


def line_chart(xlabels, series, aria, ymax=1.0, W=760, H=260, R=170, every=1, fmt=lambda v: f(v, 2)):
    """series: [(nome, valores, cor)] na mesma escala 0..ymax · rótulos diretos à direita."""
    L, T, B = 46, 16, 34
    n = len(xlabels)
    x = lambda i: L + i * (W - L - R) / max(n - 1, 1)
    y = lambda v: T + (1 - v / ymax) * (H - T - B)
    grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{f(v, 1)}</text>'
                   for v in [ymax * k / 5 for k in range(6)])
    ticks = "".join(f'<text x="{x(i):.1f}" y="{H - 12}" class="ax" text-anchor="middle">{lab}</text>'
                    for i, lab in enumerate(xlabels) if i % every == 0)
    ly = spread_labels([y(vals[-1]) + 4 for _, vals, _ in series])
    body = ""
    for (name, vals, color), yl in zip(series, ly):
        body += f'<polyline points="{" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        body += "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="9" fill="transparent"><title>{xlabels[i]} · {name}: {fmt(v)}</title></circle>'
                        for i, v in enumerate(vals))
        body += f'<text x="{x(n - 1) + 10:.1f}" y="{yl:.1f}" class="lb">{name}</text>'
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">{CSS}{grid}{ticks}{body}</svg>'


def legend(series):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, _, c in series) + "</div>"


nd = json.loads((A / "ndvi_class_medians.json").read_text())
ND_SERIES = [(k, nd["classes"][k]["median"], CAT[i]) for i, k in
             enumerate(["trigo de inverno", "cevada de inverno", "milho", "pastagem"])]


def ndvi_classes_chart():
    xl = [mlabel(d) for d in nd["dates"]]
    svg = line_chart(xl, ND_SERIES, "Curva mediana de NDVI por cultura nas parcelas de treino", every=3, H=280)
    # faixa dos estágios do trigo (BBCH aproximado), sob o eixo
    # (rótulo curto na faixa, nome completo no tooltip)
    stages = [(("semeadura", "semeadura"), "2018-10-01", "2018-11-15"),
              (("perfilhamento · repouso", "perfilhamento e repouso de inverno"), "2018-11-15", "2019-02-28"),
              (("alongam.", "alongamento do colmo"), "2019-03-01", "2019-04-20"),
              (("espigam.", "espigamento e floração"), "2019-04-20", "2019-05-31"),
              (("matur.", "enchimento de grãos e maturação"), "2019-06-01", "2019-07-10"),
              (("colh.", "colheita"), "2019-07-10", "2019-08-05")]
    from datetime import date
    d0 = date.fromisoformat(nd["dates"][0]); d1 = date.fromisoformat(nd["dates"][-1])
    W, L, R = 760, 46, 170
    xx = lambda iso: L + (date.fromisoformat(iso) - d0).days / (d1 - d0).days * (W - L - R)
    shades = ["#1c355e", "#162a4a"]
    strip = "".join(f'<rect x="{xx(a) + 1:.1f}" y="4" width="{xx(b) - xx(a) - 2:.1f}" height="22" fill="{shades[i % 2]}" rx="4">'
                    f'<title>{full}</title></rect><text x="{(xx(a) + xx(b)) / 2:.1f}" y="19" class="ax" text-anchor="middle" '
                    f'font-size="10">{short}</text>' for i, ((short, full), a, b) in enumerate(stages))
    return (svg + f'<svg viewBox="0 0 {W} 30" role="img" aria-label="Estágios fenológicos do trigo de inverno">{CSS}{strip}</svg>'
            + legend(ND_SERIES))


def learning_chart(hist, keys, aria):
    xl = [str(h["epoch"]) for h in hist]
    series = [(name, [h[k] for h in hist], CAT[i]) for i, (k, name) in enumerate(keys)]
    return line_chart(xl, series, aria, every=2, H=230) + legend(series)


def iou_bars():
    items = sorted(m1["test_iou_per_class"].items(), key=lambda kv: -kv[1])
    W, rowh, L = 760, 22, 200
    H = rowh * len(items) + 10
    bars = ""
    for i, (name, v) in enumerate(items):
        yy = 5 + i * rowh
        color = CAT[0] if name == "trigo de inverno" else CAT[1]
        w = max(v * (W - L - 60), 2)
        bars += (f'<text x="{L - 8}" y="{yy + 14}" class="ax" text-anchor="end">{name}</text>'
                 f'<rect x="{L}" y="{yy + 3}" width="{w:.1f}" height="{rowh - 6}" rx="4" fill="{color}">'
                 f'<title>{name}: IoU {f(v, 3)}</title></rect>'
                 f'<text x="{L + w + 6:.1f}" y="{yy + 14}" class="ax">{f(v, 2)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="IoU por classe no teste, Agent 01">{CSS}{bars}</svg>'


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
early = m1["early_season"]["by_cutoff"]
early_rows = "".join(row(dlabel(e["date"]) + "/" + e["date"][2:4], f(e["tau"], 3), pct(e["confirmed_share_of_wheat"]),
                         pct(e["precision_of_confirmed"]), pct(e.get("val_precision", float("nan"))), f(e["wheat_auc"], 3))
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
_c = llm[C.LLM_MODEL]
llm_row_chosen = row(f"<b>{C.LLM_MODEL}</b>", pct(_c["validation_pass_rate"]), pct(_c["risk_level_agreement"]),
                     f"{f(_c['latency_s_median'], 1)} s", f"{_c['prompt_tokens_mean']} + {_c['completion_tokens_mean']}")
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

# confusões do trigo por parcela (Agent 01, teste)
from common import LABELS_PT
ptab = json.loads((A / "agent01/parcel_preds_test.json").read_text())
fn = collections.Counter(LABELS_PT[t["pred"]] for t in ptab if t["label"] == C.CROP_ID and t["pred"] != C.CROP_ID)
fp = collections.Counter(LABELS_PT[t["label"]] for t in ptab if t["pred"] == C.CROP_ID and t["label"] != C.CROP_ID)
conf_txt = lambda c: ", ".join(f"{k} ({v})" for k, v in c.most_common(4))
ss = json.loads((A / "split_stats.json").read_text())
split_rows = "".join(row(nm, str(ss[k]["patches"]), str(ss[k]["parcels"]), str(ss[k]["wheat_parcels"]),
                         f(ss[k]["wheat_pixel_pct"], 1) + "%", " · ".join(f"{t[1:]}: {n}" for t, n in sorted(ss[k]["tiles"].items())))
                     for k, nm in (("train", "Treino (folds 1-3)"), ("val", "Validação (fold 4)"), ("test", "Teste (fold 5)")))
h1, h2 = m1["history"], m2["history"]

# ---------------------------------------------------------------------------
# Exemplos ponta a ponta (um slide por caso, gerados dos relatórios)
# ---------------------------------------------------------------------------
E2E = [
    ("10259_1147481", "Trigo honesto e saudável", "Declaração e área confirmadas; vigor no topo da região. É o caso mais comum e deve gerar risco baixo sem alarmes."),
    ("30455_50323", "Trigo real com vigor muito baixo", "Trigo de verdade, mas fraco: pico de NDVI 0,72 e integral entre as menores da região. O modelo temporal fica em dúvida (o trigo ainda passa no limiar de declaração). O risco alto vem do vigor, não de fraude, e o parecer pede vistoria."),
    ("40025_7275742", "Trigo com cobertura verde depois da colheita", "Depois da colheita (julho) a área ficou verde de novo em set-out. A fenologia procura o pico só na janela do ciclo do trigo; sem essa regra, o pico de outubro era tomado como o do trigo."),
    ("10270_1110439", "Área declarada 30% maior que a do cadastro", "Trigo de verdade, mas o polígono declarado invade o entorno. Detectado pela auditoria geométrica do Agent 02."),
    ("30455_13881546", "Milho declarado como trigo", "A cultura declarada não é confirmada pela série temporal; a comparação com os trigos da região deixa de se aplicar."),
    ("40025_13362682", "Área inflada, mas não auditável", "Parcela de 1,08 ha: pequena demais para separar a faixa externa do núcleo do polígono. A auditoria não emite pontuação; o sistema registra o motivo e não acusa sem evidência."),
]
LEVEL_PT = {"low": "baixo", "medium": "médio", "high": "alto"}
SCEN_PT = {"honesto": "declaração honesta", "area_inflada": "área declarada +30%", "cultura_falsa": "cultura falsa"}


def ev_name(key):
    for n in (f"{key}_infl", key):
        if (A / "agent02/evidence" / f"{n}.png").exists():
            return n
    return None


def e2e_slide(i, key, title, intro):
    r = reports[key]
    a1, a2, a3, a4 = r["agent01_output"], r["agent02_output"], r["agent03_output"], r["agent04_output"]
    dc, aa, rc = a3["declaration_check"], a3["area_audit"], a3["regional_comparison"]
    conf = dlabel(dc["confirmed_since"]) if dc["confirmed_since"] else "não confirmado"
    sup = {True: "sustentada", False: "<b>não sustentada</b>", None: "não auditável (" + {"parcel_cut_by_image_edge": "borda da imagem"}.get(aa.get("not_auditable_reason"), "parcela pequena demais") + ")"}[aa["declared_area_supported"]]
    score_line = (f'pontuação {f(aa["inconsistency_score"], 1)} (limiar {f(aa["threshold_inconsistency"], 1)})'
                  if aa.get("inconsistency_score") is not None else "sem pontuação")
    reg = (f'integral do NDVI no percentil {f(rc["ndvi_integral_percentile"], 1)} · pico {f(rc["peak_ndvi"])} em {dlabel(rc["peak_date"])} '
           f'({rc["peak_delay_days_vs_region"]:+d} dias vs. mediana da região)' if rc["applicable"] else "não se aplica (cultura não confirmada)")
    flags = ", ".join(x for x in a3["data_quality_flags"] if x != "vigor_not_validated_against_yield") or "—"
    recs = "".join(f'<li><b>[{x["prioridade"]}]</b> {html.escape(x["acao"])}<br><span class="accent small">↳ {x["evidencia_referenciada"]}</span></li>'
                   for x in a4["recomendacoes"])
    ev = ev_name(key)
    img = f'<img class="ev" src="{img_b64(ev, 520)}" alt="Evidência na imagem SPOT: polígono declarado em amarelo, bordas detectadas em ciano">' if ev else ""
    lv = a2["low_vigor"]["low_vigor_pct"]
    return f"""<section class="slide e2e">
  <div class="kicker">20 · Exemplo ponta a ponta {i}/{len(E2E)} · {SCEN_PT[r["scenario"]]}</div>
  <h2>{title}</h2>
  <p>{intro}</p>
  <div class="grid2">
    <div class="card"><span class="tagline">Evidências dos agentes</span>
      <p class="small"><b>Agent 01 · série temporal:</b> P(trigo) = {f(dc["p_declared"], 3)} (limiar {f(dc["threshold_tau"], 2)}) · mais provável: {a1["observed_crop"]} · confirmado desde: {conf}<br>
      <b>Agent 02 · imagem:</b> área declarada {f(aa["declared_area_ha"])} ha × cadastro {f(aa["registry_area_ha"])} ha · {score_line} → {sup} · manchas de baixo vigor: {f(lv, 1) + "%" if lv is not None else "—"}<br>
      <b>Agent 03 · região:</b> {reg}<br>
      <b>Score:</b> {f(a3["risk_score"])} → risco <b>{LEVEL_PT[a3["risk_level"]]}</b> · sinalizações: {flags}</p>
      <div class="viz">{img}
      <div class="chart">{ndvi_chart(r)}
        <div class="legend"><span><i style="background:#de6c00"></i>esta parcela</span><span><i style="background:#4f8fe6"></i>mediana regional</span><span><i style="background:#4f8fe6;opacity:.45"></i>p25–p75</span><span><i style="background:#4f8fe6;opacity:.25"></i>p10–p90</span></div>
      </div></div></div>
    <div class="card"><span class="tagline">Parecer · Agent 04 · {C.LLM_MODEL}</span><p class="small">{html.escape(a4["diagnostico"])}</p><ul class="small">{recs}</ul></div>
  </div>
</section>"""


e2e_html = "\n\n".join(e2e_slide(i + 1, *c) for i, c in enumerate(E2E) if c[0] in reports)

subs = {
    "E2E_SLIDES": e2e_html,
    "NDVI_CLASSES": ndvi_classes_chart(), "IOU_BARS": iou_bars(),
    "LC1": learning_chart(h1, [("wheat_iou", "IoU trigo (val)"), ("miou", "mIoU 18 culturas (val)")],
                          "Curva de aprendizado do Agent 01: métricas de validação por época"),
    "LC2": learning_chart(h2, [("boundary_f1", "F1 borda (val)"), ("extent_iou", "IoU extensão (val)"),
                               ("parcel_iou_median", "IoU parcela, mediana (val)")],
                          "Curva de aprendizado do Agent 02: métricas de validação por época"),
    "A1_LOSS0": f(h1[0]["loss"]), "A1_LOSS1": f(h1[-1]["loss"]), "A2_LOSS0": f(h2[0]["loss"]), "A2_LOSS1": f(h2[-1]["loss"]),
    "A1_EPOCHS": str(m1["epochs"]), "A2_EPOCHS": str(m2["epochs"]),
    "FN_TXT": conf_txt(fn), "FP_TXT": conf_txt(fp), "N_FN": str(sum(fn.values())), "N_FP": str(sum(fp.values())),
    "SPLIT_ROWS": split_rows, "A1_OA": f(pix1["overall_accuracy"]),
    "A2_VAL_BF1": f(m2["val"]["boundary_f1"]), "A2_VAL_EXT": f(m2["val"]["extent_iou"]), "A2_VAL_PIOU": f(m2["val"]["parcel_iou_median"]),
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
    "APR_DATE": dlabel(apr["date"]),
    "EARLY_PMIN": pct(min(e["precision_of_confirmed"] for e in early)),
    "EARLY_PMAX": pct(max(e["precision_of_confirmed"] for e in early)), "MAY_SHARE": pct(may["confirmed_share_of_wheat"]), "MAY_PREC": pct(may["precision_of_confirmed"]),
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
    "LLM_ROWS": llm_rows, "LLM_ROW_CHOSEN": llm_row_chosen, "LLM_N": str(n_llm), "LLM_MODEL": C.LLM_MODEL,
    "S_4O": sample_text("gpt-4o-mini"), "S_41": sample_text("gpt-4.1-mini"), "S_54": sample_text("gpt-5.4-mini"),
    "INF_DECL": f(aai["declared_area_ha"]), "INF_REG": f(aai["registry_area_ha"]), "INF_SCORE": f(aai["inconsistency_score"], 1),
    "INF_TAU": f(aai["threshold_inconsistency"], 1), "INF_P": f(a3i["declaration_check"]["p_declared"], 3),
    "INF_CONF": dlabel(a3i["declaration_check"]["confirmed_since"]) if a3i["declaration_check"]["confirmed_since"] else "—",
    "INF_SCORE_RISK": f(a3i["risk_score"]), "INF_LEVEL": a4i["nivel_risco"],
    "INF_DIAG": html.escape(a4i["diagnostico"]), "INF_RECS": recs_inf,
    "MIS_P": f(a3m["declaration_check"]["p_declared"], 3), "MIS_ALT": a3m["declaration_check"]["temporal_model_alternative"],
    "MIS_DIAG": html.escape(a4m["diagnostico"]),
    # leitura humana dos três pareceres acima (artifacts/agent04/sample_pareceres.json)
    "LLM_QUAL": ("<b>Análise de erros</b> (mesmo caso nas duas rodadas): o gpt-4o-mini lê 1,21% de manchas como "
                 "“vigor baixo” (é pouco) e, numa rodada anterior, atribuiu à imagem a área do cadastro. O gpt-4.1-mini "
                 "confunde imagem × cadastro nas duas rodadas e trata a pontuação de inconsistência como porcentagem. "
                 "O gpt-5.4-mini separa declarado × cadastro e cita o percentil regional certo. As poucas falhas "
                 "automáticas dos três são números calculados pelo próprio modelo (somas, diferenças) que não estão "
                 "no payload. Custo da escolha: ~1,5× mais tokens de saída, com latência menor."),
}
out = Path(__file__).with_name("presentation_template.html").read_text()
for k, v in subs.items():
    out = out.replace("{{" + k + "}}", v)
assert "{{" not in out, [l for l in out.splitlines() if "{{" in l][:3]
dst = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(out, encoding="utf-8")
print(f"ok → {dst} ({dst.stat().st_size / 1024:.0f} KB)")
