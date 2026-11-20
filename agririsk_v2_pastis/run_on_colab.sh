#!/usr/bin/env bash
# Treina a v2 numa GPU do Google Colab a partir deste terminal (google-colab-cli).
#
#   ./run_on_colab.sh up       # cria a VM com GPU, envia o código e inicia o treino em background
#   ./run_on_colab.sh sync     # reenvia o código e relança o treino na VM existente
#   ./run_on_colab.sh status   # fim do log de treino
#   ./run_on_colab.sh fetch    # baixa modelos, métricas e benchmark para ./artifacts
#   ./run_on_colab.sh down     # desliga a VM
#
# Variáveis: GPU (T4), PER_FOLD (30), EPOCHS01 (80), EPOCHS02 (40), COLAB_SESSION (agririsk-v2)
# Requer: colab-cli autenticado e PYTHON_KEYRING_BACKEND configurado (ver README).
set -euo pipefail
cd "$(dirname "$0")"
S="${COLAB_SESSION:-agririsk-v2}"
GPU="${GPU:-T4}"
CODE="config.py common.py prepare_data.py prepare_rpg.py agent01_temporal_utae.py agent02_field_delineation.py
      agent03_regional_fusion.py agent04_analyst.py orchestrator.py colab_remote_train.sh
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
  rm -rf "$tmp"
  cexec <<PY
import subprocess, tarfile, os
os.makedirs("/content/agririsk_v2", exist_ok=True)
tarfile.open("/content/agririsk_v2_code.tgz").extractall("/content/agririsk_v2", filter="data")
subprocess.run("pkill -f colab_remote_train.sh", shell=True)
log = open("/content/agririsk_v2/train.log", "w")
env = {**os.environ, "PER_FOLD": "${PER_FOLD:-30}", "EPOCHS01": "${EPOCHS01:-80}", "EPOCHS02": "${EPOCHS02:-40}"}
subprocess.Popen(["bash", "colab_remote_train.sh"], cwd="/content/agririsk_v2", stdout=log,
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
print("treino iniciado na VM")
PY
  echo "Acompanhe com: ./run_on_colab.sh status"
}

case "${1:-}" in
  up)     colab new -s "$S" --gpu "$GPU"; sync_and_launch ;;
  sync)   sync_and_launch ;;
  status) echo 'print(open("/content/agririsk_v2/train.log").read()[-3000:])' | cexec ;;
  fetch)
    colab download -s "$S" /content/artifacts_v2.tgz ./artifacts_v2.tgz
    tar xzf artifacts_v2.tgz && rm artifacts_v2.tgz
    ls -la artifacts/* ;;
  down)   colab stop -s "$S" ;;
  *)      sed -n '2,11p' "$0"; exit 1 ;;
esac
