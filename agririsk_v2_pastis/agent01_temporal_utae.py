"""
Agent 01 — Spatio-Temporal Crop Agent (U-TAE-lite)
AgriRisk-MA v2 · Segmentação semântica de culturas a partir da SEQUÊNCIA de
                 imagens Sentinel-2 (26 datas x 10 bandas x 128 x 128, 10 m)

Por que um modelo espaço-temporal (e não a curva média da parcela)?
  - Usa forma, vizinhança e textura (espaço) junto com a fenologia (tempo).
  - Não depende de conhecer o contorno da parcela de antemão.
  - É o estado da arte no PASTIS (U-TAE, Sainte Fare Garnot & Landrieu, ICCV 2021);
    aqui numa versão enxuta (~0,5 M parâmetros) para treinar em minutos.
  - Baseline obrigatório para justificar a escolha: Random Forest sobre a curva
    média da parcela (26 x 10), avaliado nas MESMAS parcelas de teste.

Arquitetura
  encoder conv 2D compartilhado por data (3 escalas)
  → atenção temporal multi-cabeça por pixel (L-TAE-lite) na menor escala,
    com codificação senoidal do dia de cada imagem
  → os MESMOS pesos de atenção agregam as conexões de cada escala
  → decoder U-Net → 20 classes por pixel (fundo, 18 culturas, void ignorado)

Confirmação antecipada na safra
  Treino com CORTES TEMPORAIS ALEATÓRIOS: em metade dos lotes a série é truncada numa
  data sorteada (datas posteriores mascaradas na atenção). Um único modelo responde
  "dá para confirmar o trigo com o que se viu até a data D?". Avaliação por data de
  corte: F1/AUC e % dos trigos confirmados com precisão ≥ 95% (limiar da validação).

Avaliação (fold 5 oficial do PASTIS, mesmas regiões do treino)
  pixel: OA, mIoU, IoU/F1 do trigo · parcela: F1 e AUC de "é trigo?" por voto
  majoritário / probabilidade média dentro da parcela.

Uso
  python agent01_temporal_utae.py train [--epochs 80]
  python agent01_temporal_utae.py demo
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config as C
from common import GRID_DATES, LABELS_PT, load_patch, manifest, parcel_ndvi, phenology

OUT = C.ARTIFACTS / "agent01"
DEV = C.DEVICE
AMP = dict(device_type=DEV.type, dtype=torch.float16, enabled=C.ON_GPU)


# ---------------------------------------------------------------------------
# 1. MODELO
# ---------------------------------------------------------------------------
class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(inplace=True))


def day_encoding(days: torch.Tensor, d: int) -> torch.Tensor:
    """Codificação senoidal do dia (como em Transformers), períodos até ~1 ano."""
    i = torch.arange(d // 2, device=days.device, dtype=torch.float32)
    freq = 1.0 / (1000 ** (2 * i / d))
    ang = days[:, None].float() * freq[None]
    return torch.cat([ang.sin(), ang.cos()], 1)                 # (T, d)


class UTAELite(nn.Module):
    def __init__(self, in_ch=10, widths=(32, 64, 128), heads=4, d_k=16, n_classes=C.N_CLASSES):
        super().__init__()
        w0, w1, w2 = widths
        self.heads, self.d_k = heads, d_k
        self.inc = ConvBlock(in_ch, w0)
        self.down1 = nn.Sequential(nn.Conv2d(w0, w0, 4, 2, 1), ConvBlock(w0, w1))
        self.down2 = nn.Sequential(nn.Conv2d(w1, w1, 4, 2, 1), ConvBlock(w1, w2))
        self.key = nn.Linear(w2, heads * d_k)
        self.query = nn.Parameter(torch.randn(heads, d_k) * 0.1)
        self.up2 = nn.ConvTranspose2d(w2, w1, 2, 2)
        self.dec2 = ConvBlock(2 * w1, w1)
        self.up1 = nn.ConvTranspose2d(w1, w0, 2, 2)
        self.dec1 = ConvBlock(2 * w0, w0)
        self.head = nn.Conv2d(w0, n_classes, 1)
        days = torch.tensor([(d - GRID_DATES[0]).days for d in GRID_DATES])
        self.register_buffer("pos", day_encoding(days, w2))

    def _attend(self, f, mask=None):           # f: (B, T, C, h, w) · mask: (B, T) datas válidas
        B, T, Cc, h, w = f.shape
        x = f.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, Cc) + self.pos[None]
        k = self.key(x).view(-1, T, self.heads, self.d_k)
        scores = (k * self.query).sum(-1) / math.sqrt(self.d_k)                      # (N, T, H)
        if mask is not None:
            m = mask[:, None, None, :].expand(B, h, w, T).reshape(-1, T, 1)
            scores = scores.masked_fill(~m, float("-inf"))
        att = torch.softmax(scores, dim=1)
        return att.view(B, h, w, T, self.heads).permute(0, 4, 3, 1, 2)            # (B, H, T, h, w)

    @staticmethod
    def _aggregate(f, att):                    # f (B,T,C,Hs,Ws), att (B,H,T,h,w)
        B, T, Cc, Hs, Ws = f.shape
        H = att.shape[1]
        a = F.interpolate(att.reshape(B * H, T, *att.shape[-2:]), size=(Hs, Ws), mode="bilinear",
                          align_corners=False).view(B, H, T, Hs, Ws)
        f = f.view(B, T, H, Cc // H, Hs, Ws)
        return (f * a.permute(0, 2, 1, 3, 4)[:, :, :, None]).sum(1).reshape(B, Cc, Hs, Ws)

    def forward(self, x, return_attention=False, mask=None):     # x: (B, T, 10, H, W)
        B, T = x.shape[:2]
        if mask is not None:
            x = x * mask[:, :, None, None, None]               # datas futuras zeradas
        e0 = self.inc(x.flatten(0, 1))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        unflat = lambda e: e.view(B, T, *e.shape[1:])
        att = self._attend(unflat(e2), mask)
        s2 = self._aggregate(unflat(e2), att)
        s1 = self._aggregate(unflat(e1), att)
        s0 = self._aggregate(unflat(e0), att)
        d = self.dec2(torch.cat([self.up2(s2), s1], 1))
        d = self.dec1(torch.cat([self.up1(d), s0], 1))
        logits = self.head(d)
        return (logits, att) if return_attention else logits


# ---------------------------------------------------------------------------
# 2. DADOS
# ---------------------------------------------------------------------------
def stack_split(split):
    rows = manifest(split)
    X = np.stack([load_patch(r["patch"])["s2"] for r in rows])        # (N, T, 10, H, W) f16
    Y = np.stack([load_patch(r["patch"])["target"] for r in rows]).astype(np.int64)
    return rows, X, Y


def norm_stats(X):
    x = X.astype(np.float32)
    return x.mean(axis=(0, 1, 3, 4)), x.std(axis=(0, 1, 3, 4)) + 1e-6


def to_input(x, mean, std):
    x = (x.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.from_numpy(x)


def augment(x, y, rng):
    k = int(rng.integers(4))
    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    if rng.integers(2):
        x, y = x.flip(-1), y.flip(-1)
    return x, y


# ---------------------------------------------------------------------------
# 3. MÉTRICAS
# ---------------------------------------------------------------------------
def pixel_metrics(pred, y):
    m = y != C.VOID
    p, t = pred[m], y[m]
    ious = {}
    for c in np.unique(t):
        u = ((p == c) | (t == c)).sum()
        ious[int(c)] = float(((p == c) & (t == c)).sum() / u) if u else float("nan")
    tw, pw = t == C.CROP_ID, p == C.CROP_ID
    return {"overall_accuracy": float((p == t).mean()), "miou": float(np.nanmean(list(ious.values()))),
            "wheat_iou": ious.get(C.CROP_ID, float("nan")),
            "wheat_precision": float(precision_score(tw, pw, zero_division=0)),
            "wheat_recall": float(recall_score(tw, pw, zero_division=0)),
            "wheat_f1": float(f1_score(tw, pw, zero_division=0)), "iou_per_class": ious}


def parcel_table(rows, probs):
    """probs: (N, 20, H, W) softmax. Devolve uma linha por parcela."""
    out = []
    for r, pr in zip(rows, probs):
        d = load_patch(r["patch"])
        ids = d["parcels"]
        for pid, lab, npix in zip(d["parcel_ids"], d["parcel_labels"], d["parcel_npix"]):
            if lab in (0, C.VOID) or npix < 5:
                continue
            m = ids == pid
            mean_p = pr[:, m].mean(1)
            out.append({"patch": r["patch"], "parcel": int(pid), "tile": r["tile"], "label": int(lab),
                        "npix": int(npix), "p_wheat": float(mean_p[C.CROP_ID]),
                        "pred": int(mean_p.argmax())})
    return out


def parcel_metrics(tab, key="p_wheat"):
    y = np.array([t["label"] == C.CROP_ID for t in tab])
    p = np.array([t[key] for t in tab])
    pred = np.array([t["pred"] == C.CROP_ID for t in tab]) if "pred" in tab[0] else p > 0.5
    return {"n_parcels": int(len(y)), "n_wheat": int(y.sum()),
            "wheat_f1": float(f1_score(y, pred)), "wheat_precision": float(precision_score(y, pred)),
            "wheat_recall": float(recall_score(y, pred)), "wheat_auc": float(roc_auc_score(y, p))}


CUTOFFS = [9, 11, 13, 15, 17, 19, 21, 23, C.T_STEPS - 1]     # índices da grade (jan → out/2019)


def cutoff_mask(n, cutoff):
    m = torch.zeros(n, C.T_STEPS, dtype=torch.bool, device=DEV)
    m[:, :cutoff + 1] = True
    return m


@torch.no_grad()
def predict(model, X, mean, std, batch=2, cutoff=None):
    model.eval()
    probs, atts = [], []
    for i in range(0, len(X), batch):
        xb = to_input(X[i:i + batch], mean, std).to(DEV)
        mask = cutoff_mask(len(xb), cutoff) if cutoff is not None else None
        with torch.autocast(**AMP):
            logits, att = model(xb, return_attention=True, mask=mask)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy().astype(np.float16))
        atts.append(att.float().cpu().numpy().astype(np.float16))
    return np.concatenate(probs), np.concatenate(atts)


# ---------------------------------------------------------------------------
# 4. TREINO
# ---------------------------------------------------------------------------
def train(epochs=80, batch=4, lr=1e-3):
    t0 = time.time()
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    rows_tr, Xtr, Ytr = stack_split("train")
    rows_va, Xva, Yva = stack_split("val")
    rows_te, Xte, Yte = stack_split("test")
    mean, std = norm_stats(Xtr)
    print(f"[agent01] recortes treino {len(Xtr)} · val {len(Xva)} · teste {len(Xte)} | device {DEV}")

    counts = np.bincount(Ytr[Ytr != C.VOID].ravel(), minlength=C.N_CLASSES).astype(float)
    w = np.where(counts > 0, 1 / np.sqrt(counts / counts.sum() + 1e-9), 0)
    weights = torch.tensor(w / w[w > 0].mean(), dtype=torch.float32, device=DEV)

    model = UTAELite().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = math.ceil(len(Xtr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler(enabled=C.ON_GPU)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=C.VOID)

    best, best_state, history = -1, None, []
    for ep in range(epochs):
        model.train()
        perm, tot = rng.permutation(len(Xtr)), 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            x = to_input(Xtr[idx], mean, std)
            y = torch.from_numpy(Ytr[idx])
            x, y = augment(x, y, rng)
            x, y = x.to(DEV), y.to(DEV)
            # metade dos lotes com corte temporal aleatório (confirmação antecipada)
            cuts = np.where(rng.random(len(idx)) < 0.5, rng.integers(8, C.T_STEPS, len(idx)), C.T_STEPS - 1)
            mask = torch.arange(C.T_STEPS, device=DEV)[None] <= torch.as_tensor(cuts, device=DEV)[:, None]
            opt.zero_grad()
            with torch.autocast(**AMP):
                logits = model(x, mask=mask)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            pv, _ = predict(model, Xva, mean, std)
            m = pixel_metrics(pv.argmax(1), Yva)
            pv15, _ = predict(model, Xva, mean, std, cutoff=15)              # meados de abril
            m15 = pixel_metrics(pv15.argmax(1), Yva)
            # foco no trigo (safra completa e antecipada) sem abandonar as demais culturas
            score = (m["wheat_iou"] + m["miou"] + m15["wheat_iou"]) / 3
            history.append({"epoch": ep + 1, "loss": tot / steps, **{k: m[k] for k in ("miou", "wheat_iou")}})
            print(f"  época {ep + 1:>3}/{epochs} loss {tot / steps:.3f} · val mIoU {m['miou']:.3f} "
                  f"IoU trigo {m['wheat_iou']:.3f} · {time.time() - t0:.0f}s")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_s = time.time() - t0

    # ---- teste: U-TAE-lite --------------------------------------------------
    pte, ate = predict(model, Xte, mean, std)
    pix = pixel_metrics(pte.argmax(1), Yte)
    tab_te = parcel_table(rows_te, pte)
    par = parcel_metrics(tab_te)
    pva, _ = predict(model, Xva, mean, std)
    tab_va = parcel_table(rows_va, pva)

    # ---- confirmação antecipada: limiar de precisão ≥ 95% na validação ------
    from sklearn.metrics import precision_recall_curve
    yv = np.array([t["label"] == C.CROP_ID for t in tab_va])
    pv_full = np.array([t["p_wheat"] for t in tab_va])
    prec, rec, thr = precision_recall_curve(yv, pv_full)
    ok = np.where(prec[:-1] >= 0.95)[0]
    tau_conf = float(thr[ok[0]]) if len(ok) else 0.9
    early = []
    for c in CUTOFFS:
        pc, _ = predict(model, Xte, mean, std, cutoff=c)
        tab_c = parcel_table(rows_te, pc)
        mc = parcel_metrics(tab_c)
        yt = np.array([t["label"] == C.CROP_ID for t in tab_c])
        pt = np.array([t["p_wheat"] for t in tab_c])
        conf = pt >= tau_conf
        early.append({"cutoff_step": c, "date": GRID_DATES[c].isoformat(),
                      "wheat_f1": round(mc["wheat_f1"], 4), "wheat_auc": round(mc["wheat_auc"], 4),
                      "confirmed_share_of_wheat": round(float((conf & yt).sum() / max(yt.sum(), 1)), 4),
                      "precision_of_confirmed": round(float((conf & yt).sum() / max(conf.sum(), 1)), 4)})
        print(f"  corte {GRID_DATES[c]} · F1 trigo {mc['wheat_f1']:.3f} AUC {mc['wheat_auc']:.3f} · "
              f"confirmados {early[-1]['confirmed_share_of_wheat']:.0%} (precisão {early[-1]['precision_of_confirmed']:.0%})")

    # ---- baseline: Random Forest na curva média da parcela -----------------
    def curves(rows):
        X, y, keys = [], [], []
        for r in rows:
            d = load_patch(r["patch"])
            for pid, lab, npix, cur in zip(d["parcel_ids"], d["parcel_labels"], d["parcel_npix"], d["parcel_curves"]):
                if lab in (0, C.VOID) or npix < 5:
                    continue
                X.append(cur.astype(np.float32).ravel()); y.append(int(lab)); keys.append((r["patch"], int(pid)))
        return np.array(X), np.array(y), keys
    Xc_tr, yc_tr, _ = curves(rows_tr)
    Xc_te, yc_te, keys_te = curves(rows_te)
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=C.SEED, class_weight="balanced")
    rf.fit(Xc_tr, yc_tr)
    pw = rf.predict_proba(Xc_te)[:, list(rf.classes_).index(C.CROP_ID)]
    rf_tab = [{"label": int(l), "p_wheat": float(p), "pred": int(c)}
              for l, p, c in zip(yc_te, pw, rf.predict(Xc_te))]
    rf_par = parcel_metrics(rf_tab)

    # ---- interpretabilidade: em que datas o modelo presta atenção no trigo --
    wheat_mask = Yte == C.CROP_ID
    att_mean = []
    for a, m in zip(ate, wheat_mask):          # a: (H, T, h, w) na escala 1/4
        if m.any():
            mm = torch.from_numpy(m[None, None].astype(np.float32))
            mm = F.interpolate(mm, size=a.shape[-2:], mode="area")[0, 0].numpy() > 0.5
            if mm.any():
                att_mean.append(a.astype(np.float32)[:, :, mm].mean(-1))   # (H, T)
    att_mean = np.mean(att_mean, 0) if att_mean else np.zeros((4, C.T_STEPS))

    print("\n--- Agent 01 · teste (fold 5, mesmas regiões) ---")
    print(f"  pixel  · OA {pix['overall_accuracy']:.3f} · mIoU {pix['miou']:.3f} · IoU trigo {pix['wheat_iou']:.3f}"
          f" · F1 trigo {pix['wheat_f1']:.3f}")
    print(f"  parcela· U-TAE-lite F1 trigo {par['wheat_f1']:.3f} AUC {par['wheat_auc']:.3f} "
          f"| baseline RF F1 {rf_par['wheat_f1']:.3f} AUC {rf_par['wheat_auc']:.3f} ({par['n_parcels']} parcelas)")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "utae_lite.pt")
    np.savez(OUT / "norm.npz", mean=mean, std=std)
    metrics = {
        "model": "U-TAE-lite", "params": n_params, "epochs": epochs, "device": str(DEV),
        "train_seconds": round(train_s, 1), "patches": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "test_pixel": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pix.items() if k != "iou_per_class"},
        "test_iou_per_class": {LABELS_PT[c]: round(v, 3) for c, v in pix["iou_per_class"].items()},
        "test_parcel": {k: round(v, 4) if isinstance(v, float) else v for k, v in par.items()},
        "baseline_rf_parcel": {k: round(v, 4) if isinstance(v, float) else v for k, v in rf_par.items()},
        "early_season": {"tau_confirm": round(tau_conf, 4), "target_precision": 0.95, "by_cutoff": early},
        "attention_on_wheat": {"dates": [d.isoformat() for d in GRID_DATES],
                               "per_head": np.round(att_mean, 4).tolist()},
        "history": history,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "parcel_preds_val.json").write_text(json.dumps(tab_va))
    (OUT / "parcel_preds_test.json").write_text(json.dumps(tab_te))


# ---------------------------------------------------------------------------
# 4b. CONFIRMAÇÃO ANTECIPADA COM LIMIAR POR DATA (só inferência)
# ---------------------------------------------------------------------------
def evaluate_early(target_precision=0.95):
    """Para cada data de corte: limiar escolhido na VALIDAÇÃO para precisão ≥ alvo
    (maximizando a cobertura) e medido no TESTE. Um limiar único não serve: o modelo fica
    menos confiante no início da safra.

    Limiar MONOTÔNICO: τ(data) = min(τ_validação(data), τ(data anterior)). Calibrados de forma
    independente numa validação pequena, os limiares oscilam entre datas (ex.: maio 0,52 →
    junho 0,61), e a parcela "confirmada em maio" seria "desconfirmada em junho". Com o limiar
    nunca subindo, a confirmação só cresce ao longo da safra; a precisão na validação é
    registrada para mostrar o custo disso."""
    from sklearn.metrics import precision_recall_curve
    c = _load()
    rows_va, Xva, _ = stack_split("val")
    rows_te, Xte, _ = stack_split("test")
    out, prev = [], 1.0
    for k in CUTOFFS:
        tv = parcel_table(rows_va, predict(c["model"], Xva, c["mean"], c["std"], cutoff=k)[0])
        tt = parcel_table(rows_te, predict(c["model"], Xte, c["mean"], c["std"], cutoff=k)[0])
        yv, pv = np.array([t["label"] == C.CROP_ID for t in tv]), np.array([t["p_wheat"] for t in tv])
        yt, pt = np.array([t["label"] == C.CROP_ID for t in tt]), np.array([t["p_wheat"] for t in tt])
        prec, rec, thr = precision_recall_curve(yv, pv)
        ok = np.where(prec[:-1] >= target_precision)[0]
        tau_raw = float(thr[ok[0]]) if len(ok) else 1.0
        tau = min(tau_raw, prev)
        prev = tau
        cv, conf = pv >= tau, pt >= tau
        mc = parcel_metrics(tt)
        out.append({"cutoff_step": k, "date": GRID_DATES[k].isoformat(), "tau": round(tau, 4),
                    "tau_independent": round(tau_raw, 4),
                    "val_precision": round(float((cv & yv).sum() / max(cv.sum(), 1)), 4),
                    "wheat_f1": round(mc["wheat_f1"], 4), "wheat_auc": round(mc["wheat_auc"], 4),
                    "confirmed_share_of_wheat": round(float((conf & yt).sum() / max(yt.sum(), 1)), 4),
                    "precision_of_confirmed": round(float((conf & yt).sum() / max(conf.sum(), 1)), 4)})
        print(f"  {GRID_DATES[k]} · τ {tau:.3f} (independente {tau_raw:.3f}) · confirmados "
              f"{out[-1]['confirmed_share_of_wheat']:.0%} com precisão {out[-1]['precision_of_confirmed']:.0%} "
              f"(val {out[-1]['val_precision']:.0%}) · F1 {mc['wheat_f1']:.3f} AUC {mc['wheat_auc']:.3f}")
    m = json.loads((OUT / "metrics.json").read_text())
    m["early_season"] = {"target_precision": target_precision,
                         "threshold": "por data (validação), monotônico: nunca sobe ao longo da safra",
                         "by_cutoff": out}
    (OUT / "metrics.json").write_text(json.dumps(m, indent=1))


# ---------------------------------------------------------------------------
# 5. INFERÊNCIA → CONTRATO JSON
# ---------------------------------------------------------------------------
_CACHE = {}


def _load():
    if "model" not in _CACHE:
        m = UTAELite()
        m.load_state_dict(torch.load(OUT / "utae_lite.pt", map_location="cpu"))
        n = np.load(OUT / "norm.npz")
        _CACHE.update(model=m.to(DEV).eval(), mean=n["mean"], std=n["std"], probs={})
    return _CACHE


def patch_probs(patch: int) -> np.ndarray:
    c = _load()
    if patch not in c["probs"]:
        p, _ = predict(c["model"], load_patch(patch)["s2"][None], c["mean"], c["std"], batch=1)
        c["probs"][patch] = p[0].astype(np.float32)
    return c["probs"][patch]


def confirmation_timeline(patch: int, parcel: int, crop: int = C.CROP_ID) -> dict:
    """P(cultura declarada) a cada data de corte e a 1ª data confirmada (limiar de 95% de precisão)."""
    c = _load()
    key = ("timeline", patch)
    if key not in c["probs"]:
        x = load_patch(patch)["s2"][None]
        c["probs"][key] = {k: predict(c["model"], x, c["mean"], c["std"], batch=1, cutoff=k)[0][0].astype(np.float32)
                           for k in CUTOFFS}
    m = load_patch(patch)["parcels"] == parcel
    taus = {e["cutoff_step"]: e["tau"] for e in json.loads((OUT / "metrics.json").read_text())["early_season"]["by_cutoff"]}
    tl = [{"date": GRID_DATES[k].isoformat(), "p": round(float(v[crop][m].mean()), 3), "threshold": round(taus[k], 3)}
          for k, v in c["probs"][key].items()]
    ok = [t["p"] >= t["threshold"] for t in tl]
    first = next((i for i in range(len(tl)) if all(ok[i:])), None)      # confirmado e assim permaneceu
    return {"target_precision": 0.95, "timeline": tl,
            "confirmed_since": tl[first]["date"] if first is not None else None}


def run(patch: int, parcel: int, declared_crop: int = C.CROP_ID) -> dict:
    d = load_patch(patch)
    m = d["parcels"] == parcel
    probs = patch_probs(patch)[:, m].mean(1)
    probs[[0, C.VOID]] = 0                                  # parcela agrícola: ignora fundo/void
    probs = probs / probs.sum()
    top = np.argsort(probs)[::-1][:3]
    i = int(np.where(d["parcel_ids"] == parcel)[0][0])
    curve = parcel_ndvi(d["parcel_curves"][i].astype(np.float32))
    return {
        "patch": int(patch), "parcel_id": int(parcel), "region": str(d["tile"]),
        "area_ha": round(float(m.sum() * C.PIXEL_AREA_HA_S2), 2),
        "declared_crop": LABELS_PT[declared_crop],
        "p_declared_crop": round(float(probs[declared_crop]), 3),
        "observed_crop": LABELS_PT[int(top[0])],
        "top3": [{"crop": LABELS_PT[int(k)], "prob": round(float(probs[k]), 3)} for k in top],
        "ndvi_curve": {"dates": [g.isoformat() for g in GRID_DATES], "values": [round(float(v), 3) for v in curve]},
        "phenology": phenology(curve),
        "early_confirmation": confirmation_timeline(patch, parcel, declared_crop),
        "model": "U-TAE-lite",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "early", "demo"])
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    if a.cmd == "train":
        train(epochs=a.epochs)
    elif a.cmd == "early":
        evaluate_early()
    else:
        r = manifest("test")[0]
        d = load_patch(r["patch"])
        pid = int(d["parcel_ids"][np.argmax(d["parcel_labels"] == C.CROP_ID)])
        print(json.dumps(run(r["patch"], pid), indent=1, ensure_ascii=False))
