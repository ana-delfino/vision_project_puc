# AgriRisk-MA v2 · Trigo de inverno · PASTIS-HD

Sistema multiagente que confere se um talhão **declarado como trigo** é mesmo trigo e como ele
se compara aos **trigos da mesma região**, combinando a série temporal Sentinel-2 e a imagem SPOT
de alta resolução **da mesma parcela**. O resultado é um parecer rastreável para o analista de
crédito ou seguro rural.

- **Apresentação:** [docs/AgriRisk_MA_v2_apresentacao.html](docs/AgriRisk_MA_v2_apresentacao.html) ·
  [PDF](docs/AgriRisk_MA_v2_apresentacao.pdf) · [PNG por slide](docs/slides/)
- **Versão anterior (cana, v1):** pasta raiz do repositório, inalterada.

## Arquitetura

```
START ─┬─> Agent 01 · U-TAE-lite (26 imagens Sentinel-2) ──┐
       └─> Agent 02 · U-Net SPOT 2 m + manchas de vigor ───┴─> Agent 03 · fusão + benchmark regional ─> Agent 04 · parecer (LLM)
```

| Agente | Modelo | Por quê |
|---|---|---|
| 01 · espaço-temporal | U-TAE-lite (0,56 M parâmetros): encoder por data + atenção temporal por pixel + decoder U-Net | Visão computacional sobre a sequência de imagens; versão enxuta do estado da arte no PASTIS (U-TAE, ICCV 2021) |
| 02 · alta resolução | U-Net + MobileNetV2 (trigo × resto) e detector de manchas VARI/MAD não supervisionado | Confirmação visual independente; o PASTIS não tem rótulo de anomalia |
| 03 · fusão e região | Regressão logística (stacking) + percentis entre os trigos da mesma região (folds de treino) | A fusão é aprendida e medida; o risco é relativo aos vizinhos |
| 04 · parecer | gpt-5.4-mini (escolhido entre 3) com structured outputs, evidência em `enum` e validação | Escolha justificada por avaliação objetiva e análise de erros |

## Resultados (teste = fold 5 oficial, mesmas 4 regiões do treino)

| Métrica | Valor |
|---|---|
| Agent 01 · trigo por parcela: F1 / AUC | **0,908 / 0,991** (baseline Random Forest na curva média: 0,832 / 0,987) |
| Agent 01 · pixel: IoU do trigo / mIoU 18 culturas | 0,769 / 0,395 |
| Agent 02 · trigo por parcela: F1 / AUC · pixel IoU | 0,703 / 0,929 · 0,587 |
| Fusão · AUC / F1 / recall | 0,992 / 0,906 / 0,948 (temporal sozinho: recall 0,869) |
| Conferência da declaração · recall da divergência | **97,0%** |
| Alarmes falsos em trigos reais | **2,6 a cada 100** |
| Precisão esperada com 5% de declarações falsas (premissa) | 66% |
| LLM · pareceres que passam na validação (30 parcelas) | gpt-5.4-mini **100%** · gpt-4o-mini 97% · gpt-4.1-mini 93% |
| Treino na GPU T4 do Colab | Agent 01: 11 min · Agent 02: 3,8 min |

948 parcelas de teste (153 de trigo). Métricas completas em `artifacts/agent0*/metrics.json`.

## Volumetria

150 recortes do PASTIS-HD (30 por fold, ≥ 5% de trigo, regiões em rodízio): ~3 GB baixados,
~1 GB depois do pré-processamento. O índice [data_index/wheat_patches.json](data_index/wheat_patches.json)
evita baixar todas as anotações só para escolher os recortes.

## Como reproduzir

```bash
# a partir da raiz do repositório
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r agririsk_v2_pastis/requirements.txt playwright && playwright install chromium
cd agririsk_v2_pastis

# 1) treino na GPU do Google Colab (google-colab-cli)
uv tool install google-colab-cli --with keyrings.alt
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
colab sessions                         # login (1ª vez)
./run_on_colab.sh up                   # VM T4: prepara dados, treina 01 e 02, monta o 03
./run_on_colab.sh status               # até aparecer AGRIRISK_DONE
./run_on_colab.sh fetch && ./run_on_colab.sh down

# 2) dados locais para inferência e benchmark (CPU)
python prepare_data.py --folds 1 2 3 5
python agent03_regional_fusion.py      # benchmark regional (só folds de treino) + fusão

# 3) pareceres e avaliação do LLM (chave API_KEY_GPT no .env da raiz)
python orchestrator.py --n 8
python orchestrator.py --eval-llm --n 30 --models gpt-4o-mini gpt-4.1-mini gpt-5.4-mini

# 4) apresentação (HTML gerado a partir dos JSONs) + PDF/PNG
python build_presentation.py && python export_presentation.py
```

Sem GPU, os treinos também rodam localmente (`python agent01_temporal_utae.py train` e
`python agent02_vision_spot.py train`), mas levam horas em CPU.

## Artefatos para avaliação

| Caminho | Conteúdo |
|---|---|
| `artifacts/agent01/` | modelo U-TAE-lite, normalização, métricas (pixel, parcela, baseline RF, atenção por data, histórico) e predições por parcela |
| `artifacts/agent02/` | U-Net SPOT, métricas, predições por parcela, imagens de evidência (`evidence/`) |
| `artifacts/agent03/` | benchmark regional, coeficientes da fusão, métricas de fusão e de conferência da declaração |
| `artifacts/agent04/` | comparação dos 3 LLMs e um parecer de cada modelo para o mesmo caso |
| `artifacts/reports/` | 8 relatórios ponta a ponta (saída de cada agente + validação + tempos) |
| `docs/` | apresentação HTML, PDF e PNG por slide |

## Limitações

O vigor regional não foi validado contra produtividade (o PASTIS não tem colheita). Os dados são
de uma safra, na França. As declarações falsas são simuladas. O SPOT tem uma única data, e o
contorno a 10 m cria falsas manchas na borda. As classes raras têm IoU baixo (só 90 recortes de
treino). Detalhes no último slide da apresentação.

## Dados e referências

PASTIS-HD (IGNF/CNES, licença etalab-2.0) · Sainte Fare Garnot & Landrieu, ICCV 2021 · Astruc et al.,
ECCV 2024 (OmniSat) · Resolução CMN 4.427/2015 e 5.267/2025 (fiscalização do crédito rural por
sensoriamento remoto).
