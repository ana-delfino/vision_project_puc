"""
Agent 04 — Agricultural Analyst Agent (v2 · trigo)
AgriRisk-MA v2 · Parecer técnico para o analista de crédito/seguro rural

LLM da OpenAI (padrão gpt-5.4-mini, escolhido por avaliação; configurável) com STRUCTURED OUTPUTS:
  1. JSON sempre válido no schema do parecer (json_schema, strict).
  2. Grounding por construção: `evidencia_referenciada` é um ENUM dos caminhos
     que existem no JSON de entrada.
  3. Validação pós-geração (v2): nível de risco = Agent 03; números citados existem
     no payload; se a declaração diverge, o parecer TEM de dizer isso.
  4. Nova tentativa: se a validação reprovar, os problemas voltam ao modelo e ele
     reemite o parecer completo (AGRIRISK_LLM_RETRIES, padrão 1).
  5. Telemetria por chamada: latência, tokens e tentativas (para comparar modelos).
  6. Modo offline (template) se não houver chave.

Chave: .env com API_KEY_GPT (raiz do repositório ou desta pasta).
"""
import json
import os
import re
import time
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

import config as C

for f in C.ENV_FILES:
    if f.exists():
        load_dotenv(f)
RISK_PT = {"low": "baixo", "medium": "medio", "high": "alto"}

SYSTEM_PROMPT = """\
Você é um agrônomo que apoia analistas de crédito e seguro rural. Escreve pareceres \
sobre talhões declarados como TRIGO DE INVERNO (cultura e área) a partir de evidências \
estruturadas de um pipeline de sensoriamento remoto: série temporal Sentinel-2 (modelo \
espaço-temporal), delimitação do talhão na imagem SPOT de alta resolução (2 m) e um \
benchmark com as parcelas de trigo DA MESMA REGIÃO na mesma safra (2018-19). Você só \
conhece o que está no JSON; não presuma clima, manejo, preços ou histórico.

Como ler as evidências:
- `declaration_check`: P(a cultura declarada é real) pela série temporal. \
`consistent=false` significa que a declaração NÃO foi confirmada — isso precisa abrir o \
diagnóstico, citando a cultura alternativa sugerida pelo modelo. `confirmed_since` é a \
primeira data da safra em que a cultura já estava confirmada (com 95% de precisão). \
`consistent=true` com `confirmed_since=null` NÃO é declaração contestada: a declaração passa no \
limiar, mas o sinal temporal é fraco (nunca chegou à confirmação com 95% de precisão). Diga \
isso assim; não escreva que a cultura "não foi confirmada" nesse caso.
- `area_audit`: confere se a imagem sustenta o polígono/área declarados. `declared_area_ha` é \
a área declarada e `registry_area_ha` a do cadastro. `inconsistency_score` mede bordas de outros \
talhões e áreas não cultivadas na faixa externa do polígono em relação ao núcleo; acima de \
`threshold_inconsistency` → `declared_area_supported=false` (possível área declarada a maior): \
diga isso explicitamente, citando as áreas em hectares. `declared_minus_registry_pct` é só a \
diferença percentual entre declarado e cadastro: NUNCA a compare com `threshold_inconsistency`, \
que é a escala de `inconsistency_score`. `declared_area_supported=null` = não \
auditável; o motivo está em `not_auditable_reason` (parcela cortada pela borda da imagem, ou \
parcela pequena demais para separar a faixa externa do núcleo): cite esse motivo em `limitacoes`.
- `regional_comparison`: percentis são relativos às parcelas de trigo da mesma região \
(ex.: ndvi_integral_percentile 8 = vigor acumulado abaixo de 92% dos trigos da região). \
`peak_delay_days_vs_region` > 0 = ciclo atrasado em relação à região.
- Se `regional_comparison.applicable` = false (cultura não confirmada), NÃO use os \
percentis de vigor/atraso: eles comparam com trigo e não valem para outra cultura.
- `low_vigor_pct`: % da parcela em manchas de baixo vigor na imagem SPOT.
- A prioridade das recomendações acompanha o risco: com risco baixo, nenhuma \
recomendação de prioridade alta (prefira monitoramento de rotina).
- `nivel_risco` deve seguir `regional_fusion.risk_level` (low=baixo, medium=medio, high=alto).

Regras:
- Toda recomendação aponta em `evidencia_referenciada` o caminho do JSON que a sustenta.
- Use só números do JSON. Datas no formato do JSON.
- Liste em `limitacoes` as `data_quality_flags` relevantes, em linguagem simples \
(ex.: vigor ainda não validado contra produtividade).
- Português técnico e direto, para um analista não especialista em sensoriamento \
remoto. 2 a 4 recomendações acionáveis (ex.: vistoria de campo, pedido de documentação, \
monitoramento na próxima safra).
"""


def evidence_paths(obj: Any, prefix: str = "") -> List[str]:
    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.append(p)
            paths.extend(evidence_paths(v, p))
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            paths.append(p)
            paths.extend(evidence_paths(v, p))
    return paths


def build_schema(paths: List[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["diagnostico", "nivel_risco", "parecer_risco", "recomendacoes", "limitacoes"],
        "properties": {
            "diagnostico": {"type": "string"},
            "nivel_risco": {"type": "string", "enum": ["baixo", "medio", "alto"]},
            "parecer_risco": {"type": "string"},
            "recomendacoes": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["acao", "prioridade", "evidencia_referenciada"],
                "properties": {"acao": {"type": "string"},
                               "prioridade": {"type": "string", "enum": ["baixa", "media", "alta"]},
                               "evidencia_referenciada": {"type": "string", "enum": paths}}}},
            "limitacoes": {"type": "string"},
        },
    }


def _area_audit_view(aa: dict) -> dict:
    """Sem pontuação (não auditável), o limiar não tem com o que ser comparado: removê-lo evita
    que o LLM o compare com a diferença percentual declarado × cadastro."""
    if aa.get("inconsistency_score") is None:
        return {k: v for k, v in aa.items() if k != "threshold_inconsistency"}
    return aa


def build_payload(a1: dict, a2: dict, a3: dict) -> dict:
    t = {k: v for k, v in a1.items() if k not in ("model", "ndvi_curve")}
    t["ndvi_curve_values"] = a1["ndvi_curve"]["values"]
    d = {k: v for k, v in a2.items() if k != "evidence_png"}
    d["area_audit"] = _area_audit_view(a2["area_audit"])
    return {"satellite_timeseries": t,
            "field_delineation": d,
            "regional_fusion": {**a3, "area_audit": _area_audit_view(a3["area_audit"])}}


MAX_RETRIES = int(os.environ.get("AGRIRISK_LLM_RETRIES", 1))   # novas tentativas se a validação reprovar


def _chat(client, model, messages, schema):
    # modelos de raciocínio (gpt-5*, o*) só aceitam a temperatura padrão
    extra = {} if model.startswith(("gpt-5", "o1", "o3", "o4")) else {"temperature": 0.2}
    resp = client.chat.completions.create(
        model=model, **extra, messages=messages,
        response_format={"type": "json_schema", "json_schema": {
            "name": "parecer_trigo", "strict": True, "schema": schema}},
    )
    msg = resp.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"recusado: {msg.refusal}")
    return msg.content, resp.usage


def generate_parecer(a1: dict, a2: dict, a3: dict, model: str | None = None,
                     max_retries: int | None = None) -> tuple[dict, dict]:
    """Gera o parecer; se a validação pós-geração reprovar, devolve ao modelo a lista de
    problemas e pede o parecer completo corrigido (até `max_retries` vezes). O `_meta`
    registra o resultado da 1ª tentativa, para medir o ganho da nova tentativa."""
    payload = build_payload(a1, a2, a3)
    key = os.environ.get("API_KEY_GPT")
    if not key:
        return template_parecer(payload), payload
    model = model or C.LLM_MODEL
    max_retries = MAX_RETRIES if max_retries is None else max_retries
    client, schema = openai.OpenAI(api_key=key), build_schema(evidence_paths(payload))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Emita o parecer para as evidências abaixo.\n\n<evidencias>\n"
                                            + json.dumps(payload, ensure_ascii=False) + "\n</evidencias>"}]
    t0, tok_in, tok_out, first_issues = time.time(), 0, 0, None
    for attempt in range(1, max_retries + 2):
        content, usage = _chat(client, model, messages, schema)
        tok_in, tok_out = tok_in + usage.prompt_tokens, tok_out + usage.completion_tokens
        parecer = json.loads(content)
        issues = validate(parecer, payload)["issues"]
        if first_issues is None:
            first_issues = issues
        if not issues or attempt > max_retries:
            break
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": "A validação automática reprovou o parecer:\n"
                      + "\n".join(f"- {x}" for x in issues)
                      + "\nCorrija esses pontos usando só as evidências fornecidas e emita o parecer completo de novo."}]
    parecer["_meta"] = {"model": model, "latency_s": round(time.time() - t0, 2),
                        "prompt_tokens": tok_in, "completion_tokens": tok_out,
                        "attempts": attempt, "first_pass_ok": not first_issues, "first_issues": first_issues}
    return parecer, payload


def template_parecer(payload: dict) -> dict:
    f = payload["regional_fusion"]
    dc, rc = f["declaration_check"], f["regional_comparison"]
    recs = []
    if not dc["consistent"]:
        recs.append({"acao": "Solicitar vistoria de campo ou documentação para confirmar a cultura declarada.",
                     "prioridade": "alta", "evidencia_referenciada": "regional_fusion.declaration_check.consistent"})
    if rc["ndvi_integral_percentile"] is not None and rc["ndvi_integral_percentile"] < 25:
        recs.append({"acao": "Acompanhar a produtividade: vigor acumulado abaixo da maioria dos trigos da região.",
                     "prioridade": "media", "evidencia_referenciada": "regional_fusion.regional_comparison.ndvi_integral_percentile"})
    return {"diagnostico": f"Declarado: {dc['declared_crop']} · P(confirmado) {dc['p_declared']}.",
            "nivel_risco": RISK_PT[f["risk_level"]], "parecer_risco": f"Score {f['risk_score']}.",
            "recomendacoes": recs, "limitacoes": "Parecer por template, sem LLM.", "_meta": {"model": "template"}}


def _numbers(obj) -> set:
    return {float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(obj))}


def validate(parecer: dict, payload: dict) -> Dict[str, Any]:
    issues = []
    f = payload["regional_fusion"]
    if parecer["nivel_risco"] != RISK_PT[f["risk_level"]]:
        issues.append(f"nivel_risco '{parecer['nivel_risco']}' difere do Agent 03")
    known_paths = set(evidence_paths(payload))
    for i, r in enumerate(parecer["recomendacoes"]):
        if r["evidencia_referenciada"] not in known_paths:
            issues.append(f"recomendacoes[{i}] cita evidência inexistente")
    text = " ".join([parecer["diagnostico"], parecer["parecer_risco"], parecer["limitacoes"]]
                    + [r["acao"] for r in parecer["recomendacoes"]])
    base = _numbers(payload)
    # aceita conversões de unidade legítimas: probabilidade 0,997 escrita como 99,7 (%)
    known = {round(v, 2) for k in base for v in (k, k * 100, k / 100)} | {2018.0, 2019.0}
    for n in re.findall(r"(?<![\d-])\d+[.,]\d+", text):
        v = round(float(n.replace(",", ".")), 2)
        if v not in known and not any(abs(v - k) <= 0.051 for k in known):
            issues.append(f"número '{n}' não está no payload")
    if f["area_audit"]["declared_area_supported"] is False and not re.search(r"área|area", parecer["diagnostico"] + parecer["parecer_risco"], re.I):
        issues.append("área declarada não sustentada, mas o parecer não menciona a área")
    if f["risk_level"] == "low" and any(r["prioridade"] == "alta" for r in parecer["recomendacoes"]):
        issues.append("recomendação de prioridade alta num parecer de risco baixo")
    if not f["declaration_check"]["consistent"] and not re.search(r"declar|não confirm|nao confirm|diverg",
                                                                    parecer["diagnostico"] + parecer["parecer_risco"], re.I):
        issues.append("declaração divergente não mencionada no diagnóstico/parecer")
    # erros de raciocínio encontrados na análise de erros (números certos, leitura errada)
    dc, aa = f["declaration_check"], f["area_audit"]
    frase0 = r"(?:[^.]|\.\d)"   # mesmo período (o ponto decimal não encerra a frase)
    if dc["consistent"] and (re.search(r"(cultura|declara\w*|trigo)[^.]{0,40}n[ãa]o (foi |é |está )?confirmad", text, re.I)
                             or re.search(r"(?<!n[ãa]o )(?<!n[ãa]o foi )(?<!n[ãa]o é )contestad", text, re.I)):
        issues.append("declaração consistente descrita como 'não confirmada' ou 'contestada'")
    if "?" in text:
        issues.append("parecer com pergunta ou hesitação (afirme só o que as evidências sustentam)")
    if dc["consistent"] and re.search(rf"{dc['p_declared']}{frase0}{{0,30}}abaixo do limiar", text):
        issues.append("probabilidade acima do limiar descrita como abaixo")
    if re.search(r"[^\x00-\u024F\u2000-\u206F\u2190-\u22FF\u00B0-\u00BF]", text):
        issues.append("texto com caracteres fora do alfabeto latino")
    if aa["declared_area_supported"] is None:
        # o limiar não vai no payload quando não há pontuação: citar "limiar" aqui é inventar a comparação
        # (o limiar da declaração, ex. "abaixo do limiar 0.07", é outro e pode ser citado)
        frase = r"(?:[^.]|\.\d)"   # mesmo período (o ponto decimal não encerra a frase)
        if re.search(rf"limiar de inconsist|threshold_inconsistency|%{frase}{{0,40}}(acima|abaixo|al[ée]m) do limiar"
                     rf"|(área|area|cadastro){frase}{{0,60}}(acima|abaixo|al[ée]m) do limiar", text, re.I):
            issues.append("área não auditável comparada com o limiar de inconsistência")
        if aa.get("not_auditable_reason") != "parcel_cut_by_image_edge" and re.search(r"borda da imagem|cortad[ao] pela borda", text, re.I) \
                and not re.search(r"n[ãa]o (foi |está |é )?cortad", text, re.I):
            issues.append("motivo de não auditável trocado (borda da imagem)")
    return {"ok": not issues, "issues": issues}
