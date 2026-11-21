#!/usr/bin/env bash
# Toda a computação pesada da v2 roda no Google Colab (google-colab-cli), nunca na CPU local.
#
#   ./run_on_colab.sh up       # cria a VM, envia o código (e modelos, se pedido) e inicia o JOB
#   ./run_on_colab.sh sync     # reenvia o código e relança o JOB na VM existente
#   ./run_on_colab.sh status   # fim do log
#   ./run_on_colab.sh fetch    # baixa os artefatos produzidos para ./artifacts
#   ./run_on_colab.sh down     # desliga a VM (sempre: VM ligada consome unidades de computação)
#
# JOB=train (padrão) · GPU T4 · prepara dados, treina Agents 01 e 02, monta o 03
# JOB=eval           · VM só CPU · usa os modelos já treinados: auditoria de área, confirmação
#                      antecipada, benchmark regional e evidências para o Agent 04
# JOB=llm            · VM só CPU · Agent 04 sobre artifacts/payloads (LLM_TASK=reports | eval).
#                      Envia SÓ a linha API_KEY_GPT do .env, num arquivo 600 que o job apaga ao sair.
#
# Variáveis: JOB, GPU (T4 | none), HIGH_MEM (1 = máquina com mais RAM), PER_FOLD (30),
#            EPOCHS01 (80), EPOCHS02 (40), N_PAYLOADS (60), UPLOAD_MODELS (1), COLAB_SESSION,
#            LLM_TASK (reports), N_REPORTS (8), PARCELS ("recorte_parcela ..."), LLM_MODELS
# Requer: colab-cli autenticado e PYTHON_KEYRING_BACKEND configurado (ver README).
set -euo pipefail
cd "$(dirname "$0")"
JOB="${JOB:-train}"
if [ "$JOB" = "eval" ]; then GPU="${GPU:-none}"; S="${COLAB_SESSION:-agririsk-v2-cpu}"; SCRIPT=colab_remote_eval.sh
elif [ "$JOB" = "llm" ]; then GPU="${GPU:-none}"; S="${COLAB_SESSION:-agririsk-v2-llm}"; SCRIPT=colab_remote_llm.sh
else GPU="${GPU:-T4}"; S="${COLAB_SESSION:-agririsk-v2}"; SCRIPT=colab_remote_train.sh; fi
CODE="config.py common.py prepare_data.py prepare_rpg.py agent01_temporal_utae.py agent02_field_delineation.py
      agent03_regional_fusion.py agent04_analyst.py orchestrator.py build_wheat_index.py
      colab_remote_train.sh colab_remote_eval.sh colab_remote_llm.sh
      data_index/wheat_patches.json data_index/label_names.json data_index/rpg_polygons.json.gz"

cexec() {   # colab exec com retentativas (a conexão com a VM às vezes cai)
  local code; code="$(cat)"
  for i in 1 2 3 4; do
    if printf '%s\n' "$code" | timeout 180 colab exec -s "$S" "$@"; then return 0; fi
    echo "  (conexão falhou, tentativa $i/4)" >&2; sleep 5
  done
  return 1
}

sync_and_launch() {
  tmp="$(mktemp -d)"
  tar czf "$tmp/code.tgz" $CODE
  colab upload -s "$S" "$tmp/code.tgz" /content/agririsk_v2_code.tgz
  if [ "$JOB" = "eval" ] && [ "${UPLOAD_MODELS:-1}" = "1" ]; then   # modelos já treinados na GPU
    tar czf "$tmp/models.tgz" artifacts/agent01 artifacts/agent02/unet_delineation.pt artifacts/agent02/metrics.json \
      artifacts/agent02/audit_variants.json artifacts/agent03
    colab upload -s "$S" "$tmp/models.tgz" /content/agririsk_v2_models.tgz
  fi
  if [ "$JOB" = "llm" ]; then   # evidências dos Agents 01-03 + só a chave da OpenAI
    tar czf "$tmp/models.tgz" artifacts/payloads
    colab upload -s "$S" "$tmp/models.tgz" /content/agririsk_v2_models.tgz
    envf="$(cd .. && pwd)/.env"; [ -f .env ] && envf=.env
    ( umask 077; sed -nE 's/^[[:space:]]*(export[[:space:]]+)?API_KEY_GPT[[:space:]]*=[[:space:]]*/API_KEY_GPT=/p' "$envf" > "$tmp/key.env" )
    [ -s "$tmp/key.env" ] || { echo "API_KEY_GPT não encontrada em $envf"; rm -rf "$tmp"; exit 1; }
    colab upload -s "$S" "$tmp/key.env" /content/agririsk_v2_key.env >/dev/null
  fi
  rm -rf "$tmp"
  cexec <<PY
import subprocess, tarfile, os, time
# encerra o job anterior ANTES de instalar a chave: a limpeza dele (trap EXIT) apaga o .env
subprocess.run("pkill -f colab_remote_", shell=True); time.sleep(2)
os.makedirs("/content/agririsk_v2", exist_ok=True)
tarfile.open("/content/agririsk_v2_code.tgz").extractall("/content/agririsk_v2", filter="data")
if os.path.exists("/content/agririsk_v2_models.tgz") and ("${UPLOAD_MODELS:-1}" == "1" or "$JOB" == "llm"):
    tarfile.open("/content/agririsk_v2_models.tgz").extractall("/content/agririsk_v2", filter="data")
if os.path.exists("/content/agririsk_v2_key.env"):   # chave: só para o job llm, apagada ao fim do job
    os.replace("/content/agririsk_v2_key.env", "/content/agririsk_v2/.env"); os.chmod("/content/agririsk_v2/.env", 0o600)
log = open("/content/agririsk_v2/job.log", "w")
env = {**os.environ, "PER_FOLD": "${PER_FOLD:-30}", "EPOCHS01": "${EPOCHS01:-80}", "EPOCHS02": "${EPOCHS02:-40}",
       "N_PAYLOADS": "${N_PAYLOADS:-60}", "CHECK_INDEX": "${CHECK_INDEX:-0}", "REDO": "${REDO:-}",
       "LLM_TASK": "${LLM_TASK:-reports}", "N_REPORTS": "${N_REPORTS:-8}", "PARCELS": "${PARCELS:-}",
       "LLM_MODELS": "${LLM_MODELS:-gpt-5.4-mini gpt-5.4}"}
subprocess.Popen(["bash", "$SCRIPT"], cwd="/content/agririsk_v2", stdout=log,
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
print("job $JOB iniciado na VM")
PY
  echo "Acompanhe com: JOB=$JOB ./run_on_colab.sh status"
}

new_vm() {
  args=(-s "$S")
  [ "$GPU" != "none" ] && args+=(--gpu "$GPU")
  [ "${HIGH_MEM:-0}" = "1" ] && args+=(--high-mem)
  colab new "${args[@]}"
}

case "${1:-}" in
  up)     new_vm; sync_and_launch ;;
  sync)   sync_and_launch ;;
  status) echo 'print(open("/content/agririsk_v2/job.log").read()[-3000:])' | cexec ;;
  fetch)
    colab download -s "$S" /content/artifacts_v2.tgz ./artifacts_v2.tgz
    tar xzf artifacts_v2.tgz && rm artifacts_v2.tgz
    ls -la artifacts/* ;;
  down)   colab stop -s "$S" ;;
  *)      sed -n '2,15p' "$0"; exit 1 ;;
esac
