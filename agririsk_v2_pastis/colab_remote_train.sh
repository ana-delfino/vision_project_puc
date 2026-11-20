#!/usr/bin/env bash
# Executado DENTRO da VM do Colab (disparado por run_on_colab.sh).
set -euo pipefail
trap 'echo "AGRIRISK_FAILED na linha $LINENO"' ERR
cd /content/agririsk_v2

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "sem GPU"
# O Colab já traz torch (CUDA), numpy, scikit-learn e opencv. NÃO reinstalar torch.
echo "== dependências"; pip install -q segmentation-models-pytorch openai python-dotenv langgraph

echo "== dados (PASTIS-HD, ${PER_FOLD:-30} recortes por fold)"
python -u prepare_data.py --per-fold "${PER_FOLD:-30}" --workers 12
echo "== rótulos de delimitação (RPG 2019 → 2 m)"; python -u prepare_rpg.py rasterize
echo "== agent01 (U-TAE-lite + confirmação antecipada)"; python -u agent01_temporal_utae.py train --epochs "${EPOCHS01:-80}"
echo "== agent02 (delimitação + auditoria de área)"; python -u agent02_field_delineation.py train --epochs "${EPOCHS02:-40}"
echo "== agent03 (benchmark regional + declaração)"; python -u agent03_regional_fusion.py

tar czf /content/artifacts_v2.tgz artifacts
echo "AGRIRISK_DONE"
