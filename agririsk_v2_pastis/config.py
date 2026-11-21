"""
AgriRisk-MA v2 · Trigo de inverno · PASTIS-HD

Configuração central: caminhos, orçamento de dados, protocolo de avaliação e
hiperparâmetros. Toda escolha de volumetria fica aqui.
"""
import os
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AGRIRISK_DATA", ROOT / "data"))
ARTIFACTS = Path(os.environ.get("AGRIRISK_ARTIFACTS", ROOT / "artifacts"))
INDEX = ROOT / "data_index"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = DEVICE.type == "cuda"
SEED = 42

# ---------------------------------------------------------------------------
# Dados · PASTIS-HD (Hugging Face IGNF/PASTIS-HD, licença etalab-2.0)
# ---------------------------------------------------------------------------
HF_BASE = "https://huggingface.co/datasets/IGNF/PASTIS-HD/resolve/main/"
HF_FILES = {
    "s2": "DATA_S2/S2_{p}.npy",                                          # (T, 10, 128, 128) · 10 m
    "spot": "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/SPOT6_RVB_1M00_2019_{p}.tif",  # 1280² RGB · 1 m
    "target": "ANNOTATIONS/TARGET_{p}.npy",                              # (3, 128, 128) · canal 0 = cultura
    "parcels": "ANNOTATIONS/ParcelIDs_{p}.npy",                          # (128, 128) · id da parcela
}
METADATA_URL = HF_BASE + "metadata.geojson"

CROP_ID = 2                       # "Soft winter wheat"
CROP_NAME = "trigo de inverno"
N_CLASSES = 20                    # 0 fundo, 1-18 culturas, 19 void
VOID = 19
MIN_WHEAT_FRAC = 0.05             # só recortes com ≥5% de pixels de trigo

# Protocolo: folds OFICIAIS do PASTIS. As 4 regiões (tiles Sentinel-2) aparecem
# em todos os folds → treino e teste vêm das MESMAS regiões, recortes diferentes.
FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}
PATCHES_PER_FOLD = int(os.environ.get("AGRIRISK_PATCHES_PER_FOLD", 30))   # 150 recortes no total

# ---------------------------------------------------------------------------
# Série temporal Sentinel-2 → grade temporal fixa
# ---------------------------------------------------------------------------
# Cada recorte tem 38-61 datas irregulares (e com nuvens). Reamostramos para uma
# grade fixa de 16 dias (ciclo de revisita combinado ~5 dias, composição robusta).
T_START = date(2018, 9, 17)
T_STEP_DAYS = 16
T_STEPS = 26                      # 17/09/2018 → 27/10/2019
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
B_BLUE, B_RED, B_NIR = 0, 2, 6
# Ciclo do trigo de inverno na safra 2018-19: semeadura no outono, colheita até o fim de julho.
# O pico de NDVI só é procurado dentro do ciclo: depois da colheita, coberturas ou rebrota podem
# deixar a área verde de novo (set-out) e esse pico não é do trigo.
CROP_SEASON = (date(2018, 9, 17), date(2019, 8, 5))     # janela do ciclo (limites da meia-amplitude)
CROP_PEAK_WINDOW = (date(2019, 1, 1), date(2019, 7, 31)) # onde o pico do trigo pode estar
CLOUD_BLUE_THRESHOLD = 2500       # reflectância×1e4 no azul acima disso ≈ nuvem/neblina

# ---------------------------------------------------------------------------
# SPOT 6-7 (alta resolução): nativo 1,5 m, reamostrado a 1 m no PASTIS-HD.
# Reduzimos para 2 m (640 px): perde pouco detalhe real e corta 4x o volume.
# ---------------------------------------------------------------------------
SPOT_SCALE = 2                    # 1280 → 640 px
SPOT_PX = 1280 // SPOT_SCALE
SPOT_GSD_M = 1.0 * SPOT_SCALE
LABEL_UPSAMPLE = 10 // SPOT_SCALE # rótulos de 10 m → grade de 2 m (×5)

PIXEL_AREA_HA_S2 = 0.01           # 10 m × 10 m

# ---------------------------------------------------------------------------
# Agent 04 · LLM (OpenAI); chave em .env (raiz do repositório) como API_KEY_GPT
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("AGRIRISK_LLM_MODEL", "gpt-5.4-mini")   # escolhido pela avaliação (artifacts/agent04/llm_eval.json)
ENV_FILES = [ROOT / ".env", ROOT.parent / ".env"]
