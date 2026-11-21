"""
Orquestrador — AgriRisk-MA v2 (LangGraph)

    START ─┬─> Agent 01 (U-TAE-lite · série Sentinel-2: o que e desde quando) ─┐
           └─> Agent 02 (delimitação SPOT 2 m: quanto está plantado) ──────────┴─> Agent 03 (região + risco) ─> Agent 04 (parecer) ─> END

As duas modalidades olham a MESMA parcela (dados co-registrados; polígono do RPG).

Cenários de teste (todas as parcelas do fold 5, "declaradas como trigo"):
  honesto       · trigo, área do cadastro
  cultura_falsa · outra cultura declarada como trigo
  area_inflada  · trigo com área declarada 30% maior que a do cadastro

Uso
  python orchestrator.py --export-payloads --n 60       # (VM Colab) Agents 01-03 → artifacts/payloads
  python orchestrator.py --from-payloads --n 8          # (local) Agent 04 → artifacts/reports
  python orchestrator.py --eval-llm --n 40              # (local) comparação de LLMs sobre as evidências
  python orchestrator.py --n 4                          # pipeline completo num lugar só (LangGraph)
"""
import argparse
import json
import operator
import time
from typing import Annotated, Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

import config as C
import agent01_temporal_utae as a01
import agent02_field_delineation as a02
from agent03_regional_fusion import fuse_agent_outputs
from agent04_analyst import RISK_PT, generate_parecer, validate
from common import load_patch, manifest, to_jsonable


class State(TypedDict, total=False):
    patch: int
    parcel: int
    declared_area_ha: Optional[float]
    llm_model: str
    agent01_output: Optional[dict]
    agent02_output: Optional[dict]
    agent03_output: Optional[dict]
    agent04_output: Optional[dict]
    validation: Optional[dict]
    errors: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


def node(name, fn):
    def run(s):
        t0 = time.time()
        try:
            out = fn(s)
        except Exception as e:
            out = {f"{name}_output": None, "errors": [{"agent": name, "error": f"{type(e).__name__}: {e}"}]}
        out["trace"] = [{"agent": name, "seconds": round(time.time() - t0, 2)}]
        return out
    return run


def n01(s):
    return {"agent01_output": a01.run(s["patch"], s["parcel"])}


def n02(s):
    return {"agent02_output": a02.run(s["patch"], s["parcel"], s.get("declared_area_ha"))}


def n03(s):
    if not s.get("agent01_output") or not s.get("agent02_output"):
        raise RuntimeError("evidência incompleta (Agent 01 ou 02 falhou)")
    return {"agent03_output": fuse_agent_outputs(s["agent01_output"], s["agent02_output"])}


def n04(s):
    if not s.get("agent03_output"):
        return {"agent04_output": {"diagnostico": "Evidência insuficiente.", "nivel_risco": "indeterminado",
                                   "parecer_risco": "", "recomendacoes": [], "limitacoes": str(s.get("errors"))}}
    p, payload = generate_parecer(s["agent01_output"], s["agent02_output"], s["agent03_output"], s.get("llm_model"))
    return {"agent04_output": p, "validation": validate(p, payload)}


def build_pipeline():
    g = StateGraph(State)
    for n, f in [("agent01", n01), ("agent02", n02), ("agent03", n03), ("agent04", n04)]:
        g.add_node(n, node(n, f))
    g.add_edge(START, "agent01")
    g.add_edge(START, "agent02")
    g.add_edge(["agent01", "agent02"], "agent03")
    g.add_edge("agent03", "agent04")
    g.add_edge("agent04", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Seleção de parcelas de teste: trigo real + declarações falsas simuladas
# ---------------------------------------------------------------------------
INFLATION = 0.30


def pick_parcels(n, mismatch_share=0.2, inflated_share=0.2, min_ha=1.0):
    """Parcelas de teste 'declaradas como trigo' em três cenários (ver docstring do módulo)."""
    from agent02_field_delineation import load_labels
    rng = np.random.default_rng(C.SEED)
    wheat, other = [], []
    for r in manifest("test"):
        d, lab = load_patch(r["patch"]), load_labels(r["patch"])
        ids = lab["ids"]
        edge = set(np.unique(np.r_[ids[0], ids[-1], ids[:, 0], ids[:, -1]]).tolist())
        surf = dict(zip(lab["poly_ids"].tolist(), lab["surf_ha"].tolist()))
        for pid, cls in zip(d["parcel_ids"].tolist(), d["parcel_labels"].tolist()):
            if cls in (0, C.VOID) or pid in edge or surf.get(pid, 0) < min_ha:
                continue
            (wheat if cls == C.CROP_ID else other).append((r["patch"], pid, cls, surf[pid]))
    k_mis, k_inf = round(n * mismatch_share), round(n * inflated_share)
    w = [wheat[i] for i in rng.choice(len(wheat), n - k_mis, replace=False)]
    o = [other[i] for i in rng.choice(len(other), k_mis, replace=False)]
    out = [(p, q, c, None, "honesto") for p, q, c, _ in w[k_inf:]]
    out += [(p, q, c, round(a * (1 + INFLATION), 2), "area_inflada") for p, q, c, a in w[:k_inf]]
    out += [(p, q, c, None, "cultura_falsa") for p, q, c, _ in o]
    return out


def run_many(n, model=None, out_dir=None):
    pipe = build_pipeline()
    out_dir = out_dir or C.ARTIFACTS / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for patch, parcel, true_lab, decl_ha, scenario in pick_parcels(n):
        s = pipe.invoke({"patch": patch, "parcel": parcel, "declared_area_ha": decl_ha,
                         "llm_model": model or C.LLM_MODEL, "errors": [], "trace": []})
        rep = to_jsonable({k: s.get(k) for k in ("patch", "parcel", "agent01_output", "agent02_output",
                                                 "agent03_output", "agent04_output", "validation", "errors", "trace")})
        rep["true_crop_id"], rep["scenario"] = true_lab, scenario
        (out_dir / f"{patch}_{parcel}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        results.append(rep)
    return results


# ---------------------------------------------------------------------------
# Divisão de computação: Agents 01-03 (visão) na VM do Colab → evidências em JSON;
# Agent 04 (chamada de API) aqui, lendo esses JSONs. A chave da OpenAI não sai daqui.
# ---------------------------------------------------------------------------
PAYLOADS = C.ARTIFACTS / "payloads"


def export_payloads(n):
    """Roda os Agents 01-03 (com tempos) e grava uma evidência por parcela. Sem LLM."""
    PAYLOADS.mkdir(parents=True, exist_ok=True)
    for patch, parcel, lab, decl_ha, scenario in pick_parcels(n):
        t0 = time.time(); a1 = a01.run(patch, parcel); t1 = time.time()
        a2 = a02.run(patch, parcel, decl_ha); t2 = time.time()
        a3 = fuse_agent_outputs(a1, a2); t3 = time.time()
        rec = to_jsonable({"patch": patch, "parcel": parcel, "scenario": scenario, "true_crop_id": lab,
                           "declared_area_ha": decl_ha, "agent01_output": a1, "agent02_output": a2, "agent03_output": a3,
                           "trace": [{"agent": "agent01", "seconds": round(t1 - t0, 2)},
                                     {"agent": "agent02", "seconds": round(t2 - t1, 2)},
                                     {"agent": "agent03", "seconds": round(t3 - t2, 3)}]})
        (PAYLOADS / f"{scenario}_{patch}_{parcel}.json").write_text(json.dumps(rec, ensure_ascii=False))
    print(f"[export] {len(list(PAYLOADS.glob('*.json')))} evidências em {PAYLOADS}")


def load_payloads(n=None):
    files = sorted(PAYLOADS.glob("*.json"))
    recs = [json.loads(f.read_text()) for f in files]
    return recs[:n] if n else recs


def reports_from_payloads(n, model=None, parcels=None):
    """Agent 04 sobre as evidências exportadas → relatórios ponta a ponta.
    parcels: lista "recorte_parcela" para gerar casos específicos em vez do sorteio."""
    out_dir = C.ARTIFACTS / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    recs = load_payloads()
    by_s = {}
    for r in recs:
        by_s.setdefault(r["scenario"], []).append(r)
    pick = []
    for sc, k in (("honesto", n - 2 * (n // 4)), ("area_inflada", n // 4), ("cultura_falsa", n // 4)):
        rows = by_s.get(sc, [])
        pick += [rows[i] for i in rng.choice(len(rows), min(k, len(rows)), replace=False)]
    if parcels:
        pick = [r for r in recs if f"{r['patch']}_{r['parcel']}" in set(parcels)]
    for r in pick:
        t0 = time.time()
        p, payload = generate_parecer(r["agent01_output"], r["agent02_output"], r["agent03_output"], model)
        r["agent04_output"], r["validation"] = p, validate(p, payload)
        r["trace"].append({"agent": "agent04", "seconds": round(time.time() - t0, 2)})
        r["errors"] = []
        (out_dir / f"{r['patch']}_{r['parcel']}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    return pick


def eval_llm(n, models):
    """Mesmas evidências (Agents 01-03 já rodaram na VM) → cada modelo emite o parecer."""
    pipe_inputs = [(r["agent01_output"], r["agent02_output"], r["agent03_output"], r["scenario"])
                   for r in load_payloads(n)]
    summary, samples = {}, {}
    for model in models:
        rows = []
        for a1, a2, a3, lab in pipe_inputs:   # lab = cenário
            try:
                p, payload = generate_parecer(a1, a2, a3, model)
                v = validate(p, payload)
                if (model, lab) not in samples and lab != "honesto":   # 1º caso de cada cenário
                    samples[(model, lab)] = {"model": model, "scenario": lab, "patch": a1["patch"],
                                             "parcel": a1["parcel_id"], **p}
                rows.append({"ok": v["ok"], "issues": v["issues"], "risk_match": p["nivel_risco"] == RISK_PT[a3["risk_level"]],
                             "n_recs": len(p["recomendacoes"]), **p["_meta"]})
            except Exception as e:
                rows.append({"ok": False, "issues": [f"erro: {e}"], "risk_match": False, "n_recs": 0,
                             "latency_s": None, "prompt_tokens": 0, "completion_tokens": 0,
                             "attempts": 0, "first_pass_ok": False, "first_issues": [f"erro: {e}"]})
        lat = [r["latency_s"] for r in rows if r["latency_s"]]
        summary[model] = {
            "n": len(rows),
            "first_pass_rate": round(float(np.mean([r["first_pass_ok"] for r in rows])), 3),
            "validation_pass_rate": round(float(np.mean([r["ok"] for r in rows])), 3),
            "retry_rate": round(float(np.mean([r["attempts"] > 1 for r in rows])), 3),
            "risk_level_agreement": round(float(np.mean([r["risk_match"] for r in rows])), 3),
            "latency_s_median": round(float(np.median(lat)), 2) if lat else None,
            "prompt_tokens_mean": round(float(np.mean([r["prompt_tokens"] for r in rows]))),
            "completion_tokens_mean": round(float(np.mean([r["completion_tokens"] for r in rows]))),
            "first_issues": [i for r in rows for i in r["first_issues"]][:12],
            "issues": [i for r in rows for i in r["issues"]][:12],
        }
        print(f"{model:<16} 1ª tentativa {summary[model]['first_pass_rate']:.0%} → final {summary[model]['validation_pass_rate']:.0%} · risco igual ao Agent 03 "
              f"{summary[model]['risk_level_agreement']:.0%} · latência {summary[model]['latency_s_median']}s · "
              f"tokens {summary[model]['prompt_tokens_mean']}+{summary[model]['completion_tokens_mean']}")
    out = C.ARTIFACTS / "agent04"
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_eval.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "sample_pareceres.json").write_text(json.dumps(list(samples.values()), indent=1, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--eval-llm", action="store_true")
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5.4-mini"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--parcels", nargs="+", default=None, help="(local) casos específicos: recorte_parcela")
    ap.add_argument("--export-payloads", action="store_true", help="(VM) Agents 01-03 → artifacts/payloads")
    ap.add_argument("--from-payloads", action="store_true", help="(local) Agent 04 sobre as evidências exportadas")
    a = ap.parse_args()
    if a.export_payloads:
        export_payloads(a.n)
    elif a.eval_llm:
        eval_llm(a.n, a.models)
    else:
        for r in (reports_from_payloads(a.n, a.model, a.parcels) if a.from_payloads else run_many(a.n, a.model)):
            a3, a4 = r["agent03_output"], r["agent04_output"]
            print(f"\n=== recorte {r['patch']} · parcela {r['parcel']} · cenário {r['scenario']} (classe real {r['true_crop_id']}) ===")
            if a3:
                dc, rc = a3["declaration_check"], a3["regional_comparison"]
                aa = a3["area_audit"]
                print(f"  área decl {aa['declared_area_ha']} ha (cadastro {aa['registry_area_ha']}) ok={aa['declared_area_supported']} · "
                      f"confirmado desde {dc['confirmed_since']}")
                print(f"  declaração ok={dc['consistent']} P={dc['p_declared']} · vigor pct "
                      f"{rc['ndvi_integral_percentile']} · atraso {rc['peak_delay_days_vs_region']}d · "
                      f"manchas {rc['low_vigor_pct']}% → risco {a3['risk_level']} ({a3['risk_score']})")
            print(f"  parecer: {a4['nivel_risco']} · {len(a4['recomendacoes'])} recs · validação "
                  f"{(r.get('validation') or {}).get('ok')} · {r['trace']} · erros {r['errors']}")
