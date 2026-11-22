"""
Agent 02 — Field Delineation & Area Audit Agent (SPOT 6-7, 2 m)
AgriRisk-MA · "Quanto está plantado?" — o detalhe espacial que só a alta resolução dá

Por que este agente (e não classificar a cultura na imagem)?
  Identificar a cultura é um problema TEMPORAL (Agent 01). Uma imagem única de alta
  resolução brilha em GEOMETRIA: bordas, carreadores, cercas, áreas não cultivadas.
  No crédito rural, a área entra no cálculo do valor financiado →
  área declarada maior que a cultivada é sobrefinanciamento.

Rótulos: polígonos declarados do RPG 2019 (IGN/ASP) rasterizados a 2 m (prepare_rpg.py).

Modelo multitarefa (estilo ResUNet-a, Waldner & Diakogiannis 2020)
  U-Net + MobileNetV2 (ImageNet) com 2 saídas sigmoid por pixel:
    extensão (pixel pertence a alguma parcela declarada) e borda (limite entre parcelas).
  Instâncias: sementes = extensão > 0,5 e borda < 0,5 → componentes conexos →
  watershed sobre a probabilidade de borda, restrito à extensão.

Auditoria de área
  área delimitada (instância que melhor casa com o polígono declarado) ÷ área declarada.
  Limiar τ_área escolhido na VALIDAÇÃO para 5% de alarmes em declarações honestas;
  no TESTE mede-se a detecção de declarações infladas em +15% e +30% (simuladas).

Manchas de baixo vigor (não supervisionado, mantido da versão anterior): VARI dentro do
polígono declarado a 2 m, limiar robusto mediana − 2·MAD, área mínima 100 m².

Uso
  python agent02_field_delineation.py train [--epochs 40]
  python agent02_field_delineation.py demo
"""
import argparse
import json
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from common import load_patch, manifest
from prepare_rpg import LABEL_DIR, boundaries

OUT = C.ARTIFACTS / "agent02"
DEV = C.DEVICE
AMP = dict(device_type=DEV.type, dtype=torch.float16, enabled=C.ON_GPU)
CROP = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PX_HA = C.SPOT_GSD_M ** 2 / 10_000
MIN_PARCEL_HA = 0.5
BOUNDARY_TOL_PX = 2              # tolerância de 4 m na métrica de borda
INFLATIONS = (0.15, 0.30)
MAD_K, MIN_SPOT_M2 = 2.0, 100


# ---------------------------------------------------------------------------
# 1. DADOS
# ---------------------------------------------------------------------------
def load_labels(patch):
    d = np.load(LABEL_DIR / f"{patch}.npz")
    return {k: d[k] for k in d.files}


def targets(ids):
    return np.stack([(ids > 0), boundaries(ids)]).astype(np.uint8)        # (2, H, W)


def load_split(split):
    rows = manifest(split)
    X = np.stack([load_patch(r["patch"])["spot"] for r in rows])
    Y = np.stack([targets(load_labels(r["patch"])["ids"]) for r in rows])
    return rows, X, Y


def to_tensor(img):
    return torch.from_numpy((img.astype(np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)


def random_batch(X, Y, rng, batch=16):
    xs, ys = [], []
    for _ in range(batch):
        i = int(rng.integers(len(X)))
        r, c = rng.integers(0, C.SPOT_PX - CROP + 1, 2)
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        im = np.rot90(X[i, r:r + CROP, c:c + CROP], k)
        lb = np.rot90(Y[i, :, r:r + CROP, c:c + CROP], k, axes=(1, 2))
        if flip:
            im, lb = im[:, ::-1], lb[:, :, ::-1]
        im = np.clip(im.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
        xs.append(to_tensor(np.ascontiguousarray(im)))
        ys.append(torch.from_numpy(np.ascontiguousarray(lb)).float())
    return torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)


# ---------------------------------------------------------------------------
# 2. MODELO E LOSS
# ---------------------------------------------------------------------------
def build_model(pretrained=True):
    return smp.Unet("mobilenet_v2", encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=2, decoder_channels=(128, 64, 32, 16, 16))


class MultiTaskLoss(nn.Module):
    """BCE + Dice para extensão e borda; a borda é rara → pos_weight."""

    def __init__(self, pos_weight_boundary):
        super().__init__()
        self.register_buffer("pw", torch.tensor([1.0, pos_weight_boundary]).view(1, 2, 1, 1))

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)
        p = torch.sigmoid(logits)
        inter = (p * y).sum((0, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)
        return bce + dice.mean()


@torch.no_grad()
def predict(model, X, batch=4):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.stack([to_tensor(im) for im in X[i:i + batch]]).to(DEV)
        with torch.autocast(**AMP):
            p = torch.sigmoid(model(x).float())
        out.append(p.cpu().numpy().astype(np.float16))
    return np.concatenate(out)                                             # (N, 2, H, W)


# ---------------------------------------------------------------------------
# 3. INSTÂNCIAS (watershed) E MÉTRICAS
# ---------------------------------------------------------------------------
def instances(prob):
    """prob: (2, H, W) [extensão, borda] → mapa de instâncias (int32, 0 = fora)."""
    ext, bnd = prob[0].astype(np.float32), prob[1].astype(np.float32)
    field = ext > 0.5
    seeds = cv2.erode((field & (bnd < 0.5)).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, markers = cv2.connectedComponents(seeds)
    markers = markers.astype(np.int32)
    markers[~field] = n + 1                                                # fundo como marcador próprio
    relief = cv2.cvtColor((np.clip(bnd, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(relief, markers)
    markers[markers == n + 1] = 0
    # as linhas do watershed (-1) ficam sem dono: atribui ao vizinho, senão a área sai ~5% menor
    lines = markers < 0
    markers[lines] = 0
    grown = cv2.dilate(markers.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    markers[lines] = grown[lines]
    markers[~field] = 0
    # remove instâncias minúsculas (< 0,1 ha)
    ids, counts = np.unique(markers, return_counts=True)
    for i, c in zip(ids, counts):
        if i and c * PX_HA < 0.1:
            markers[markers == i] = 0
    return markers


def boundary_f1(pred_b, gt_b, tol=BOUNDARY_TOL_PX):
    if not gt_b.any() or not pred_b.any():
        return 0.0, 0.0, 0.0
    d_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    d_pr = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((d_gt[pred_b] <= tol).mean())
    r = float((d_pr[gt_b] <= tol).mean())
    return p, r, 2 * p * r / max(p + r, 1e-9)


def overlap_table(ids, inst):
    """IoU entre todas as parcelas declaradas e todas as instâncias, via tabela de contingência."""
    m = (ids > 0) & (inst > 0)
    pairs, inter = np.unique(np.stack([ids[m], inst[m]]), axis=1, return_counts=True)
    a_ids, a_cnt = np.unique(ids[ids > 0], return_counts=True)
    b_ids, b_cnt = np.unique(inst[inst > 0], return_counts=True)
    size_a, size_b = dict(zip(a_ids.tolist(), a_cnt.tolist())), dict(zip(b_ids.tolist(), b_cnt.tolist()))
    best = {}
    for (a, b), n in zip(pairs.T.tolist(), inter.tolist()):
        iou = n / (size_a[a] + size_b[b] - n)
        if iou > best.get(a, (0, 0.0, 0))[1]:
            best[a] = (b, iou, size_b[b])
    return best, size_a


def parcel_matches(lab, inst, min_ha=MIN_PARCEL_HA):
    """Para cada parcela declarada inteira no recorte: melhor instância, IoU e áreas."""
    ids = lab["ids"]
    H = ids.shape[0]
    best, size_a = overlap_table(ids, inst)
    edge = set(np.unique(np.r_[ids[0], ids[-1], ids[:, 0], ids[:, -1]]).tolist())   # cortadas pela borda
    out = []
    for pid, surf in zip(lab["poly_ids"].tolist(), lab["surf_ha"].tolist()):
        if pid in edge or size_a.get(pid, 0) * PX_HA < min_ha:
            continue
        b, iou, nb = best.get(pid, (0, 0.0, 0))
        out.append({"parcel": int(pid), "declared_ha": float(surf), "raster_ha": float(size_a[pid] * PX_HA),
                    "iou": float(iou), "delineated_ha": float(nb * PX_HA)})
    return out


def area_ratio(m):
    """Área delimitada ÷ área declarada. Sem instância casada → 0."""
    return m["delineated_ha"] / max(m["declared_ha"], 1e-6)


# ---------------------------------------------------------------------------
# 3b. AUDITORIA DE ÁREA POR EVIDÊNCIA GEOMÉTRICA
# ---------------------------------------------------------------------------
# A instância delimitada não é confiável para medir área: parcelas vizinhas do mesmo
# produtor, sem borda visível, fundem-se. Em vez disso, perguntamos se o POLÍGONO
# DECLARADO engloba o que não caberia num único talhão: bordas fortes no interior
# (cruza o limite de outro talhão) ou área não cultivada (estrada, mata, construção).
AUDIT_ERODE_PX = 3          # ignora a faixa de 6 m junto ao contorno declarado


def inflate_mask(mask, k):
    """Amplia o polígono declarado em +k de área (escala √(1+k) em torno do centroide)."""
    ys, xs = np.where(mask)
    cy, cx, sc = ys.mean(), xs.mean(), np.sqrt(1 + k)
    M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
    return cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)


RING_FRAC = 0.30             # faixa externa = 30% da área do polígono
VARIANTS = {
    "A": "% de borda no interior + % não cultivada (polígono inteiro)",
    "B": "(borda + não cultivada) na faixa externa − o mesmo no núcleo",
    "C": "(borda + não cultivada) só na faixa externa",
}


def ring_core(mask, frac=RING_FRAC):
    """núcleo = erosão até sobrar (1 − frac) da área; faixa = resto, sem a borda de 6 m."""
    area, core, k = mask.sum(), mask.copy(), 0
    while core.sum() > (1 - frac) * area and k < 80:
        core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        k += 1
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * AUDIT_ERODE_PX + 1,) * 2, np.uint8)).astype(bool)
    return core, inner & ~core


def audit_features(prob, mask, variant=None):
    """Componentes e as 3 pontuações; `inconsistency_score` = a da variante em uso."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * AUDIT_ERODE_PX + 1,) * 2, np.uint8)).astype(bool)
    core, ring = ring_core(mask)
    if inner.sum() < 50 or core.sum() < 30 or ring.sum() < 30:
        return None
    ext, bnd = prob[0].astype(np.float32), prob[1].astype(np.float32)
    bad = lambda r: 100 * ((bnd[r] > 0.5).mean() + (ext[r] < 0.5).mean())
    f = {"interior_boundary_pct": round(float(100 * (bnd[inner] > 0.5).mean()), 2),
         "non_field_pct": round(float(100 * (ext[mask] < 0.5).mean()), 2),
         "outer_ring_score": round(float(bad(ring)), 2), "core_score": round(float(bad(core)), 2)}
    f["score_A"] = round(f["interior_boundary_pct"] + f["non_field_pct"], 2)
    f["score_B"] = round(f["outer_ring_score"] - f["core_score"], 2)
    f["score_C"] = f["outer_ring_score"]
    f["inconsistency_score"] = f[f"score_{variant or current_variant()}"]
    return f


def current_variant():
    p = OUT / "audit_variants.json"
    return json.loads(p.read_text())["chosen_on_validation"] if p.exists() else "A"


def _audit_rows(rows, probs, inflations=(0.0,) + INFLATIONS, min_ha=MIN_PARCEL_HA):
    out = []
    for r, pr in zip(rows, probs):
        ids = load_labels(r["patch"])["ids"]
        H = ids.shape[0]
        for pid in np.unique(ids[ids > 0]):
            m = ids == pid
            if m.sum() * PX_HA < min_ha:
                continue
            for k in inflations:
                mk = inflate_mask(m, k) if k else m
                ys, xs = np.where(mk)
                if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == H - 1:
                    break                                      # sai da imagem: não auditável
                f = audit_features(pr, mk, "A")
                if f:
                    out.append({"patch": r["patch"], "parcel": int(pid), "inflation": k, **f})
    return out


def _calibrate(va, te, key):
    """τ = p95 das declarações honestas da validação (5% de alarmes); métricas no teste."""
    from sklearn.metrics import roc_auc_score
    hv = np.array([x[key] for x in va if x["inflation"] == 0])
    tau = float(np.percentile(hv, 95))
    out = {"tau": round(tau, 3)}
    for split, rows in (("val", va), ("test", te)):
        h = np.array([x[key] for x in rows if x["inflation"] == 0])
        out[f"{split}_false_alarm"] = round(float((h > tau).mean()), 4)
        for k in INFLATIONS:
            sk = np.array([x[key] for x in rows if x["inflation"] == k])
            out[f"{split}_det_{int(k * 100)}"] = round(float((sk > tau).mean()), 4)
            out[f"{split}_auc_{int(k * 100)}"] = round(float(roc_auc_score(
                np.r_[np.zeros(len(h)), np.ones(len(sk))], np.r_[h, sk])), 4)
    out["n_val"], out["n_test"] = int((np.array([x["inflation"] for x in va]) == 0).sum()), \
        int((np.array([x["inflation"] for x in te]) == 0).sum())
    return out


def audit():
    """Compara as 3 variantes, ESCOLHE pela validação (maior detecção de +30% com 5% de
    alarmes; desempate pela AUC) e reporta o teste. Só inferência."""
    c = _load()
    rows_va, rows_te = manifest("val"), manifest("test")
    pv = predict(c["model"], np.stack([load_patch(r["patch"])["spot"] for r in rows_va]))
    pt = predict(c["model"], np.stack([load_patch(r["patch"])["spot"] for r in rows_te]))
    va, te = _audit_rows(rows_va, pv), _audit_rows(rows_te, pt)
    variants = {v: _calibrate(va, te, f"score_{v}") for v in VARIANTS}
    for v, r in variants.items():
        print(v, VARIANTS[v], json.dumps(r), flush=True)
    best = max(VARIANTS, key=lambda v: (variants[v]["val_det_30"], variants[v]["val_auc_30"]))
    (OUT / "audit_variants.json").write_text(json.dumps(
        {"description": VARIANTS, "results": variants, "chosen_on_validation": best}, indent=1, ensure_ascii=False))
    b = variants[best]
    res = {"variant": best, "method": VARIANTS[best], "tau_inconsistency": b["tau"], "n_test_parcels": b["n_test"],
           "honest_false_alarm_rate_test": b["test_false_alarm"]}
    for k in INFLATIONS:
        res[f"detection_rate_inflated_{int(k * 100)}pct"] = b[f"test_det_{int(k * 100)}"]
        res[f"auc_inflated_{int(k * 100)}pct"] = b[f"test_auc_{int(k * 100)}"]
    print("escolhida pela validação:", best, json.dumps(res, ensure_ascii=False))
    m = json.loads((OUT / "metrics.json").read_text())
    if "area_audit" in m and "variant" not in m["area_audit"]:
        m.setdefault("area_audit_instance_based_discarded", m["area_audit"])
    m["area_audit"] = res
    (OUT / "metrics.json").write_text(json.dumps(m, indent=1))
    (OUT / "area_audit_test.json").write_text(json.dumps(te))


# ---------------------------------------------------------------------------
# 4. MANCHAS DE BAIXO VIGOR (dentro do polígono declarado a 2 m)
# ---------------------------------------------------------------------------
def vari(img):
    r, g, b = [img[..., i].astype(np.float32) for i in range(3)]
    return (g - r) / (g + r - b + 1e-3)


def declared_mask(patch, parcel, erode_px=1):
    m = (load_labels(patch)["ids"] == parcel).astype(np.uint8)
    return cv2.erode(m, np.ones((2 * erode_px + 1,) * 2, np.uint8)).astype(bool) if erode_px else m.astype(bool)


def low_vigor_spots(img, mask):
    v = np.clip(vari(img), -1, 1)
    vals = v[mask]
    if vals.size < 50:
        return {"low_vigor_pct": None, "n_spots": 0, "largest_spot_ha": 0.0}, np.zeros_like(mask)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-4
    anom = cv2.morphologyEx(((v < med - MAD_K * mad) & mask).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(anom, connectivity=8)
    keep, sizes = np.zeros_like(mask), []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] * C.SPOT_GSD_M ** 2 >= MIN_SPOT_M2:
            keep |= lab == k
            sizes.append(int(stats[k, cv2.CC_STAT_AREA]))
    return {"low_vigor_pct": round(float(100 * keep.sum() / mask.sum()), 2), "n_spots": len(sizes),
            "largest_spot_ha": round(max(sizes, default=0) * PX_HA, 3)}, keep


# ---------------------------------------------------------------------------
# 5. TREINO E AVALIAÇÃO
# ---------------------------------------------------------------------------
def evaluate(rows, probs):
    ext_i = ext_u = 0
    bf, matches = [], []
    for r, pr in zip(rows, probs):
        lab = load_labels(r["patch"])
        gt_ext = lab["ids"] > 0
        p_ext = pr[0].astype(np.float32) > 0.5
        ext_i += (gt_ext & p_ext).sum(); ext_u += (gt_ext | p_ext).sum()
        bf.append(boundary_f1(pr[1].astype(np.float32) > 0.5, boundaries(lab["ids"])))
        inst = instances(pr)
        for m in parcel_matches(lab, inst):
            matches.append({"patch": r["patch"], **m})
    bf = np.array(bf)
    ious = np.array([m["iou"] for m in matches])
    return {"extent_iou": float(ext_i / max(ext_u, 1)),
            "boundary_precision": float(bf[:, 0].mean()), "boundary_recall": float(bf[:, 1].mean()),
            "boundary_f1": float(bf[:, 2].mean()),
            "n_parcels": len(matches), "parcel_iou_median": float(np.median(ious)),
            "parcels_recovered_iou50": float((ious >= 0.5).mean())}, matches


def area_audit_metrics(matches_val, matches_te):
    rv = np.array([area_ratio(m) for m in matches_val])
    tau = float(np.percentile(rv, 5))                         # 5% de alarmes em declarações honestas (val)
    rt = np.array([area_ratio(m) for m in matches_te])
    ok = np.array([m["iou"] >= 0.5 for m in matches_te])
    out = {"tau_area_ratio": round(tau, 4),
           "honest_false_alarm_rate_test": round(float((rt < tau).mean()), 4),
           "area_abs_pct_error_median_matched": round(float(np.median(np.abs(rt[ok] - 1)) * 100), 2)
           if ok.any() else None}
    for k in INFLATIONS:
        out[f"detection_rate_inflated_{int(k * 100)}pct"] = round(float((rt / (1 + k) < tau).mean()), 4)
    return out


def train(epochs=40, lr=1e-3):
    t0 = time.time()
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    rows_tr, Xtr, Ytr = load_split("train")
    rows_va, Xva, Yva = load_split("val")
    rows_te, Xte, Yte = load_split("test")
    steps = max(len(Xtr) * 6 // 16, 1)
    print(f"[agent02] SPOT treino {len(Xtr)} · val {len(Xva)} · teste {len(Xte)} | device {DEV}")
    bfrac = Ytr[:, 1].mean()
    model = build_model().to(DEV)
    loss_fn = MultiTaskLoss(float(min(np.sqrt((1 - bfrac) / bfrac), 10))).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler(enabled=C.ON_GPU)

    best, best_state, history = -1, None, []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            x, y = random_batch(Xtr, Ytr, rng)
            opt.zero_grad()
            with torch.autocast(**AMP):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            m, _ = evaluate(rows_va, predict(model, Xva))
            score = (m["boundary_f1"] + m["parcel_iou_median"]) / 2
            history.append({"epoch": ep + 1, "loss": tot / steps, **{k: m[k] for k in ("boundary_f1", "parcel_iou_median", "extent_iou")}})
            print(f"  época {ep + 1:>3}/{epochs} loss {tot / steps:.3f} · val F1 borda {m['boundary_f1']:.3f} "
                  f"IoU parcela {m['parcel_iou_median']:.3f} · {time.time() - t0:.0f}s")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_s = time.time() - t0

    m_va, matches_va = evaluate(rows_va, predict(model, Xva))
    m_te, matches_te = evaluate(rows_te, predict(model, Xte))
    audit = area_audit_metrics(matches_va, matches_te)
    spots = []
    for r in rows_te:
        lab = load_labels(r["patch"])
        d = load_patch(r["patch"])
        for pid, lb in zip(d["parcel_ids"], d["parcel_labels"]):
            if lb == C.CROP_ID and (lab["ids"] == pid).sum() * PX_HA >= MIN_PARCEL_HA:
                s, _ = low_vigor_spots(d["spot"], declared_mask(r["patch"], pid))
                if s["low_vigor_pct"] is not None:
                    spots.append(s["low_vigor_pct"])
    print("\n--- Agent 02 · teste (fold 5, mesmas regiões) ---")
    print(f"  extensão IoU {m_te['extent_iou']:.3f} · borda F1 {m_te['boundary_f1']:.3f} (tolerância 4 m)")
    print(f"  parcelas: IoU mediana {m_te['parcel_iou_median']:.3f} · recuperadas (IoU≥0,5) "
          f"{m_te['parcels_recovered_iou50']:.0%} de {m_te['n_parcels']}")
    print(f"  auditoria de área: {json.dumps(audit)}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "unet_delineation.pt")
    metrics = {"model": "U-Net + MobileNetV2 (ImageNet) · extensão + borda + watershed", "epochs": epochs,
               "device": str(DEV), "train_seconds": round(train_s, 1), "gsd_m": C.SPOT_GSD_M,
               "labels": "RPG 2019 (IGN/ASP) rasterizado a 2 m",
               "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_te.items()},
               "val": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_va.items()},
               "area_audit": audit,
               "low_vigor_pct_wheat_test": {"median": round(float(np.median(spots)), 2),
                                            "p90": round(float(np.percentile(spots, 90)), 2), "n": len(spots)},
               "history": history}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "parcel_matches_test.json").write_text(json.dumps(matches_te))


# ---------------------------------------------------------------------------
# 6. INFERÊNCIA → CONTRATO JSON + EVIDÊNCIA VISUAL
# ---------------------------------------------------------------------------
_CACHE = {}


def _load():
    if "model" not in _CACHE:
        m = build_model(pretrained=False)
        m.load_state_dict(torch.load(OUT / "unet_delineation.pt", map_location="cpu"))
        aa = json.loads((OUT / "metrics.json").read_text()).get("area_audit", {})
        _CACHE.update(model=m.to(DEV).eval(), probs={}, inst={}, tau=aa.get("tau_inconsistency", float("inf")))
    return _CACHE


def run(patch: int, parcel: int, declared_area_ha: float | None = None, save_evidence=True) -> dict:
    """declared_area_ha: área declarada a auditar. Se maior que a do cadastro (RPG), o
    polígono é ampliado proporcionalmente (simula a geometria que teria aquela área)."""
    c = _load()
    d, lab = load_patch(patch), load_labels(patch)
    if patch not in c["probs"]:
        c["probs"][patch] = predict(c["model"], d["spot"][None])[0].astype(np.float32)
    prob = c["probs"][patch]
    rpg_mask = lab["ids"] == parcel
    i = np.where(lab["poly_ids"] == parcel)[0]
    rpg_ha = float(lab["surf_ha"][i[0]]) if len(i) else float(rpg_mask.sum() * PX_HA)
    declared = float(declared_area_ha) if declared_area_ha is not None else rpg_ha
    k = max(0.0, declared / max(rpg_ha, 1e-6) - 1)
    decl_mask = inflate_mask(rpg_mask, k) if k > 0.01 else rpg_mask
    ys, xs = np.where(decl_mask)
    edge = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == C.SPOT_PX - 1 or xs.max() == C.SPOT_PX - 1)
    f = audit_features(prob, decl_mask)
    supported = None if (f is None or edge) else bool(f["inconsistency_score"] <= c["tau"])
    reason = "parcel_cut_by_image_edge" if edge else "parcel_too_small_for_ring_core" if f is None else None
    spots, spot_mask = low_vigor_spots(d["spot"], declared_mask(patch, parcel))
    out = {
        "patch": int(patch), "parcel_id": int(parcel),
        "area_audit": {"declared_area_ha": round(declared, 2), "registry_area_ha": round(rpg_ha, 2),
                       "declared_minus_registry_pct": round(100 * k, 1),
                       **(f or {}), "threshold_inconsistency": round(c["tau"], 2),
                       "declared_area_supported": supported, "parcel_cut_by_image_edge": edge,
                       "not_auditable_reason": reason},
        "low_vigor": spots, "image": {"sensor": "SPOT 6-7", "gsd_m": C.SPOT_GSD_M},
    }
    if save_evidence:
        pad = 15
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, C.SPOT_PX)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, C.SPOT_PX)
        img, ov = d["spot"].copy(), d["spot"].copy()
        bnd = prob[1] > 0.5
        ov[bnd] = (0.4 * ov[bnd] + 0.6 * np.array([0, 220, 255])).astype(np.uint8)    # bordas detectadas (ciano)
        ov[spot_mask] = (0.35 * ov[spot_mask] + 0.65 * np.array([230, 30, 30])).astype(np.uint8)
        cn, _ = cv2.findContours(decl_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cn, -1, (255, 210, 0), 1)                          # polígono declarado (amarelo)
        crop = np.hstack([img[y0:y1, x0:x1], ov[y0:y1, x0:x1]])
        scale = max(1, 320 // max(crop.shape[0], 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ev = OUT / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        path = ev / f"{patch}_{parcel}{'_infl' if k > 0.01 else ''}.png"
        cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        out["evidence_png"] = str(path.relative_to(C.ROOT))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "audit", "demo"])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "train":
        train(epochs=a.epochs)
    elif a.cmd == "audit":
        audit()
    else:
        r = manifest("test")[0]
        d = load_patch(r["patch"])
        pid = int(d["parcel_ids"][np.argmax(d["parcel_labels"] == C.CROP_ID)])
        print(json.dumps(run(r["patch"], pid), indent=1, ensure_ascii=False))
