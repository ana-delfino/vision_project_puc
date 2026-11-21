#!/usr/bin/env bash
# Executado DENTRO de uma VM do Colab (só CPU; JOB=llm em run_on_colab.sh): Agent 04.
# Lê as evidências já exportadas (artifacts/payloads) e chama a API da OpenAI.
# A chave chega em .env (permissão 600), enviado só para este job, e é apagada ao sair,
# mesmo em caso de erro.
set -euo pipefail
cd /content/agririsk_v2
trap 'rm -f /content/agririsk_v2/.env' EXIT
trap 'echo "AGRIRISK_FAILED na linha $LINENO"' ERR

echo "== dependências"; pip install -q segmentation-models-pytorch openai python-dotenv langgraph
[ -f .env ] || { echo "AGRIRISK_FAILED: .env com API_KEY_GPT não foi enviado"; exit 1; }

case "${LLM_TASK:-reports}" in
  eval)
    echo "== Agent 04 · comparação de LLMs (${LLM_MODELS})"
    python -u orchestrator.py --eval-llm --n "${N_PAYLOADS:-60}" --models ${LLM_MODELS}
    OUT="artifacts/agent04" ;;
  reports)
    echo "== Agent 04 · relatórios ponta a ponta"
    rm -rf artifacts/reports; mkdir -p artifacts/reports
    [ "${N_REPORTS:-8}" != "0" ] && python -u orchestrator.py --from-payloads --n "${N_REPORTS:-8}"
    [ -n "${PARCELS:-}" ] && python -u orchestrator.py --from-payloads --parcels ${PARCELS}
    OUT="artifacts/reports" ;;
esac
rm -f .env
tar czf /content/artifacts_v2.tgz "$OUT"
echo "AGRIRISK_DONE"
