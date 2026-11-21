"""
Figuras estáticas do README (docs/figures/*.png), geradas só a partir de dados já baixados e de
artefatos já calculados: nenhuma inferência ou treino roda aqui.

    python build_readme_figures.py
"""
import json
from datetime import date

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from common import GRID_DATES, LABELS_PT, load_patch

A = C.ARTIFACTS
OUT = C.ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# paleta categórica validada para fundo claro (dataviz validate_palette.js --mode light):
# ordem fixa azul, laranja, água, amarelo; tons abaixo de 3:1 de contraste levam rótulo direto
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

EXAMPLE_PATCH, EXAMPLE_PARCEL = 10259, 1147481          # trigo honesto (fold 5)
LOW_PARCEL = "30455_50323"                              # trigo real de vigor muito baixo


def dl(d):
    return f"{d.day}/{MONTHS[d.month - 1]}"


def month_axis(ax, dates):
    idx = [i for i, d in enumerate(dates) if d.day <= 16 and d.month % 2 == 1]
    ax.set_xticks(idx, [f"{MONTHS[dates[i].month - 1]}/{str(dates[i].year)[2:]}" for i in idx])


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("ok →", OUT / name)


# ---------------------------------------------------------------------------
# 1. Entradas: série Sentinel-2, SPOT, polígonos RPG e rótulos
# ---------------------------------------------------------------------------
def fig_inputs():
    d = load_patch(EXAMPLE_PATCH)
    lab = np.load(C.DATA / "rpg_labels" / f"{EXAMPLE_PATCH}.npz")
    s2 = d["s2"].astype(np.float32)
    rgb = s2[:, [C.B_RED, 1, C.B_BLUE]]                     # B4, B3, B2 (cor natural)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    steps = [3, 7, 11, 13, 15, 18, 21]
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, len(steps), height_ratios=[1, 2.1], hspace=0.18, wspace=0.05)
    for j, k in enumerate(steps):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.clip((rgb[k].transpose(1, 2, 0) - lo) / (hi - lo), 0, 1))
        ax.set_title(dl(GRID_DATES[k]), fontsize=9, fontweight="normal", loc="center")
        ax.axis("off")
    fig.text(0.125, 0.955, "Sentinel-2 · 10 m · 26 datas na safra (7 mostradas, cor natural)",
             fontsize=11, fontweight="bold", color=INK)

    spot = d["spot"]
    ids = lab["ids"]
    edge = np.zeros_like(ids, np.uint8)
    for k in (1, 0):
        edge |= (np.roll(ids, 1, axis=k) != ids).astype(np.uint8)
    ov = spot.copy()
    ov[edge > 0] = (255, 220, 0)
    ov[ids == EXAMPLE_PARCEL] = (0.55 * ov[ids == EXAMPLE_PARCEL] + 0.45 * np.array([235, 104, 52])).astype(np.uint8)

    t = d["target"]
    tgt = np.full(t.shape + (3,), 235, np.uint8)
    tgt[(t > 0) & (t < C.VOID)] = (190, 190, 186)
    tgt[t == C.CROP_ID] = (42, 120, 214)
    tgt = cv2.resize(tgt, (C.SPOT_PX, C.SPOT_PX), interpolation=cv2.INTER_NEAREST)

    panels = [(spot, "SPOT 6-7 · 2 m (uma data, 2019)"),
              (ov, "Polígonos declarados (RPG 2019)\nem laranja: a parcela do exemplo"),
              (tgt, "Rótulos PASTIS · 10 m\nazul: trigo de inverno · cinza: outras culturas")]
    sub = gs[1, :].subgridspec(1, 3, wspace=0.06)
    for j, (im, title) in enumerate(panels):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(im)
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")
    save(fig, "01_entradas.png")


# ---------------------------------------------------------------------------
# 2. Por que a série temporal: NDVI mediano por cultura
# ---------------------------------------------------------------------------
def fig_phenology():
    m = json.loads((A / "ndvi_class_medians.json").read_text())
    dates = GRID_DATES
    fig, ax = plt.subplots(figsize=(10, 4.2))
    order = [("trigo de inverno", BLUE), ("cevada de inverno", ORANGE), ("milho", AQUA), ("pastagem", YELLOW)]
    ends = []
    for name, col in order:
        v = m["classes"][name]["median"]
        ax.plot(range(len(v)), v, color=col, lw=2)
        ends.append([v[-1], name, col])
    ends.sort()
    for i in range(1, len(ends)):                       # afasta rótulos diretos encavalados
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07)
    for y, name, col in ends:
        ax.text(len(dates) - 0.6, y, name, color=INK, va="center", fontsize=9.5)
    ax.axvspan(13, 18, color=BLUE, alpha=0.06, lw=0)
    ax.text(15.5, 0.06, "pico do trigo\n(abr-jun)", ha="center", color=INK2, fontsize=8.5)
    month_axis(ax, dates)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(dates) + 4)
    ax.set_ylabel("NDVI (mediana por cultura)")
    ax.set_title("O que distingue as culturas é o ciclo: trigo e cevada ficam verdes no inverno; o milho só no verão")
    save(fig, "02_ndvi_por_cultura.png")


# ---------------------------------------------------------------------------
# 3. Agent 01: confirmação antecipada ao longo da safra
# ---------------------------------------------------------------------------
def fig_early():
    e = json.loads((A / "agent01/metrics.json").read_text())["early_season"]["by_cutoff"]
    x = list(range(len(e)))
    share = [100 * r["confirmed_share_of_wheat"] for r in e]
    prec = [100 * r["precision_of_confirmed"] for r in e]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, prec, color=ORANGE, lw=2, marker="o", ms=6)
    ax.plot(x, share, color=BLUE, lw=2, marker="o", ms=6)
    for i, (s, p) in enumerate(zip(share, prec)):
        ax.text(i, s - 8, f"{s:.0f}%", ha="center", color=INK, fontsize=9)
    ax.text(len(x) - 0.85, prec[-1], "precisão dos confirmados", color=INK, va="center", fontsize=9.5)
    ax.text(len(x) - 0.85, share[-1], "trigos já confirmados", color=INK, va="center", fontsize=9.5)
    ax.set_xticks(x, [dl(GRID_DATES[r["cutoff_step"]]) for r in e])
    ax.set_xlim(-0.4, len(x) + 1.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% (teste, fold 5)")
    ax.set_xlabel("imagens disponíveis até a data")
    ax.set_title("Agent 01 · com as imagens até 13/abr, 75% dos trigos já são confirmados com ~90% de precisão")
    save(fig, "03_agent01_confirmacao_antecipada.png")


# ---------------------------------------------------------------------------
# 4. Curvas de treino (validação) dos dois modelos de visão
# ---------------------------------------------------------------------------
def fig_learning():
    h1 = json.loads((A / "agent01/metrics.json").read_text())["history"]
    h2 = json.loads((A / "agent02/metrics.json").read_text())["history"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, h, series, title in (
        (axs[0], h1, [("wheat_iou", "IoU do trigo", BLUE), ("miou", "mIoU (18 culturas)", ORANGE)],
         "Agent 01 · U-TAE-lite (80 épocas, T4, 12 min)"),
        (axs[1], h2, [("extent_iou", "IoU da extensão", BLUE), ("boundary_f1", "F1 de borda (4 m)", ORANGE)],
         "Agent 02 · U-Net MobileNetV2 (40 épocas, T4, 5 min)")):
        ep = [r["epoch"] for r in h]
        labels = []
        for key, name, col in series:
            v = [r[key] for r in h]
            ax.plot(ep, v, color=col, lw=2)
            labels.append([v[-1], f"{name}: {v[-1]:.2f}".replace(".", ",")])
        labels.sort()
        if labels[1][0] - labels[0][0] < 0.09:               # afasta rótulos diretos próximos
            labels[0][0] -= 0.045; labels[1][0] += 0.045
        for y, txt in labels:
            ax.text(ep[-1] * 1.02, y, txt, color=INK, va="center", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, ep[-1] * 1.75)
        ax.set_xlabel("época")
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("métrica na validação (fold 4)")
    save(fig, "04_curvas_de_treino.png")


# ---------------------------------------------------------------------------
# 5. Agent 02: delimitação e auditoria de área (evidências já geradas na VM)
# ---------------------------------------------------------------------------
def fig_delineation():
    ev = A / "agent02/evidence"
    cases = [("10259_1147481", "Área declarada = cadastro · pontuação 7,0 < limiar 29,3 → sustentada"),
             ("10270_1110439_infl", "Área declarada +30% (6,81 × 5,24 ha) · pontuação 59,8 > limiar 29,3 → não sustentada")]
    W = 1100
    ims = []
    for name, _ in cases:
        im = cv2.cvtColor(cv2.imread(str(ev / f"{name}.png")), cv2.COLOR_BGR2RGB)
        ims.append(cv2.resize(im, (W, int(im.shape[0] * W / im.shape[1])), interpolation=cv2.INTER_AREA))
    fig, axs = plt.subplots(2, 1, figsize=(10, 10 * sum(i.shape[0] for i in ims) / W + 1.2),
                            gridspec_kw={"height_ratios": [i.shape[0] for i in ims], "hspace": 0.12})
    for ax, im, (_, title) in zip(axs, ims, cases):
        ax.imshow(im)
        ax.set_title(title, fontsize=10.5)
        ax.axis("off")
    fig.text(0.125, 0.07, "Em cada par: à esquerda a imagem SPOT; à direita o polígono declarado (amarelo), as bordas previstas "
             "pelo modelo (ciano)\ne as manchas de baixo vigor (vermelho). Na área inflada, bordas de outros talhões "
             "atravessam a faixa externa do polígono.", fontsize=9.5, color=INK2)
    save(fig, "05_agent02_delimitacao_auditoria.png")


# ---------------------------------------------------------------------------
# 6. Agent 03: parcela × trigos da mesma região
# ---------------------------------------------------------------------------
def fig_benchmark():
    bench = json.loads((A / "agent03/regional_benchmark.json").read_text())
    rep = json.loads((A / "reports" / f"{LOW_PARCEL}.json").read_text())
    a1, a3 = rep["agent01_output"], rep["agent03_output"]
    P = bench["regions"][a3["region_benchmark_used"]]["ndvi_percentiles"]
    v = a1["ndvi_curve"]["values"]
    x = np.arange(len(v))
    rc = a3["regional_comparison"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(x, P["p10"], P["p90"], color=BLUE, alpha=0.12, lw=0)
    ax.fill_between(x, P["p25"], P["p75"], color=BLUE, alpha=0.22, lw=0)
    ax.plot(x, P["p50"], color=BLUE, lw=2)
    ax.plot(x, v, color=ORANGE, lw=2)
    ax.text(len(v) - 0.6, P["p50"][-1], "mediana dos trigos\nda região (faixas p25-p75\ne p10-p90)", color=INK, va="center", fontsize=9)
    ax.text(len(v) - 0.6, v[-1], "esta parcela", color=INK, va="center", fontsize=9.5)
    month_axis(ax, GRID_DATES)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(v) + 4.5)
    ax.set_ylabel("NDVI")
    ax.set_title(f"Agent 03 · trigo real de vigor baixo: integral do NDVI no percentil {rc['ndvi_integral_percentile']:.1f}, "
                 f"pico {rc['peak_ndvi']:.2f} em {dl(date.fromisoformat(rc['peak_date']))}".replace(".", ","))
    save(fig, "06_agent03_benchmark_regional.png")


if __name__ == "__main__":
    fig_inputs()
    fig_phenology()
    fig_early()
    fig_learning()
    fig_delineation()
    fig_benchmark()
