#!/usr/bin/env bash
# Executado DENTRO de uma VM do Colab (só CPU; JOB=eval em run_on_colab.sh).
# Usa os modelos já treinados na GPU: nenhuma etapa aqui treina pesos.
set -euo pipefail
trap 'echo "AGRIRISK_FAILED na linha $LINENO"' ERR
cd /content/agririsk_v2

echo "== máquina"; nproc; free -g | head -2
# torch da imagem do Colab funciona em CPU; NÃO reinstalar torch.
echo "== dependências"; pip install -q segmentation-models-pytorch openai python-dotenv langgraph

echo "== dados (mesma seleção determinística do treino)"
python -u prepare_data.py --per-fold "${PER_FOLD:-30}" --workers 8
python -u prepare_rpg.py rasterize
echo "== (verificação) índice de trigo reproduzível?"; python -u build_wheat_index.py --check || echo "AVISO: índice difere"
# etapas já concluídas nesta VM não são refeitas ao relançar (FORCE=1 refaz tudo)
done_() { [ "${FORCE:-0}" != "1" ] && [ -f "$1" ] && grep -q "$2" "$1"; }
echo "== agent02 · auditoria de área (3 variantes, escolha na validação)"
done_ artifacts/agent02/audit_variants.json chosen_on_validation && echo "(já feito)" || python -u agent02_field_delineation.py audit
echo "== agent01 · confirmação antecipada com limiar por data"
done_ artifacts/agent01/metrics.json '"threshold": "por data' && echo "(já feito)" || python -u agent01_temporal_utae.py early
echo "== agent03 · benchmark regional + limiar de declaração"
done_ artifacts/agent03/declaration.json tau && echo "(já feito)" || python -u agent03_regional_fusion.py
echo "== evidências dos Agents 01-03 para o Agent 04 (sem LLM)";      python -u orchestrator.py --export-payloads --n "${N_PAYLOADS:-60}"

tar czf /content/artifacts_v2.tgz artifacts/agent01/metrics.json artifacts/agent02/metrics.json \
    artifacts/agent02/audit_variants.json artifacts/agent02/area_audit_test.json artifacts/agent02/evidence \
    artifacts/agent03 artifacts/payloads
echo "AGRIRISK_DONE"
