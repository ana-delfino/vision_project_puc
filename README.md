# AgriRisk-MA: segmentação espaço-temporal e delimitação de talhões em imagens de satélite para auditoria de declarações agrícolas

#### Aluna: [Ana Paula Santos Delfino](https://github.com/ana-delfino).
#### Orientador: Leonardo Mendoza.

---

Trabalho apresentado ao curso [Visão Computacional Master (VC Master)](https://ica.ele.puc-rio.br/cursos/computer-vision-master/) da PUC-Rio como pré-requisito para conclusão de curso.

- [Link para o código](agririsk_v2_pastis/).
- [Link para a apresentação técnica (HTML)](agririsk_v2_pastis/docs/AgriRisk_MA_v2_apresentacao.html).
- Trabalhos relacionados:
    - [Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks (PASTIS, U-TAE)](https://arxiv.org/abs/2107.07933).
    - [OmniSat: Self-Supervised Modality Fusion for Earth Observation (PASTIS-HD)](https://arxiv.org/abs/2404.08351).
    - [Deep learning on edge: extracting field boundaries from satellite images with a convolutional neural network (ResUNet-a)](https://doi.org/10.1016/j.rse.2020.111741).

---

### Resumo

Em operações de crédito e seguro rural, o produtor declara a cultura e a área de cada talhão, e esses valores determinam o montante financiado ou segurado. Este trabalho propõe o AgriRisk-MA, um sistema multiagente que usa visão computacional para auditar essas declarações a partir de imagens de satélite, tendo o trigo de inverno como caso de estudo. O núcleo do sistema são dois modelos de deep learning treinados no benchmark público PASTIS-HD (França, safra 2018-19). O primeiro (Agent 01) é um U-TAE-lite, rede U-Net com atenção temporal por pixel, que segmenta 18 culturas a partir de uma sequência de 26 imagens multiespectrais Sentinel-2 (10 m). Treinado com cortes temporais aleatórios, o mesmo modelo também confirma a cultura antes do fim da safra. O segundo (Agent 02) é uma U-Net multitarefa com encoder MobileNetV2 que segmenta a extensão cultivada e as bordas entre talhões numa imagem SPOT 6-7 de alta resolução (2 m). Os rótulos de borda vêm dos polígonos declarados do Registro Parcelar Gráfico francês (RPG 2019). Com base nessas bordas, uma auditoria geométrica verifica se o polígono declarado engloba mais do que um talhão. Um terceiro agente compara cada parcela com os trigos da mesma região, e um quarto gera um parecer em linguagem natural com evidências rastreáveis. Com divisão espacial oficial do PASTIS e limiares escolhidos apenas na validação, o Agent 01 atinge F1 de 0,892 na identificação de trigo por parcela (contra 0,849 de um Random Forest) e confirma 75% dos trigos até meados de abril com cerca de 90% de precisão. O Agent 02 obtém IoU de 0,87 na extensão cultivada e F1 de 0,76 nas bordas (tolerância de 4 m), e a auditoria detecta 86,5% das áreas declaradas infladas em 30%, com 5,9% de alarmes falsos. O trabalho também discute limitações: o rótulo administrativo nem sempre corresponde a uma fronteira visível, as fraudes são simuladas e há uma única safra e uma única região geográfica.

### Abstract

In rural credit and crop insurance, farmers declare the crop and the area of each field, and these values determine the amount financed or insured. This work proposes AgriRisk-MA, a multi-agent system that uses computer vision to audit such declarations from satellite imagery, with winter wheat as the case study. Its core consists of two deep learning models trained on the public PASTIS-HD benchmark (France, 2018-19 season). The first (Agent 01) is U-TAE-lite, a U-Net with per-pixel temporal attention that segments 18 crop types from a sequence of 26 multispectral Sentinel-2 images (10 m). Trained with random temporal cut-offs, the same model also confirms the crop before the end of the season. The second (Agent 02) is a multi-task U-Net with a MobileNetV2 encoder that segments cultivated extent and field boundaries in a high-resolution SPOT 6-7 image (2 m), with boundary labels derived from declared polygons of the French Land Parcel Identification System (RPG 2019). A geometric audit built on these boundaries checks whether the declared polygon covers more than one field. A third agent compares each parcel with wheat fields in the same region, and a fourth produces a natural-language report with traceable evidence. Using the official spatial folds of PASTIS and thresholds chosen on validation only, Agent 01 reaches an F1 of 0.892 for parcel-level wheat identification (versus 0.849 for a Random Forest) and confirms 75% of wheat parcels by mid-April at about 90% precision. Agent 02 obtains an IoU of 0.87 for cultivated extent and a boundary F1 of 0.76 (4 m tolerance), and the audit detects 86.5% of declared areas inflated by 30%, at a 5.9% false-alarm rate. Limitations are discussed: administrative labels do not always match visible boundaries, the fraud scenarios are simulated, and the data cover a single season and a single country.

---

### Estrutura do repositório

O projeto está em [`agririsk_v2_pastis/`](agririsk_v2_pastis/): código, artefatos (modelos, métricas, relatórios), apresentação técnica e [README técnico](agririsk_v2_pastis/README.md) com instruções de reprodução.

### Dados

| Dado | Uso | Fonte e licença |
|---|---|---|
| **PASTIS-HD**: 2.433 recortes de 1,28 km; série Sentinel-2 (38-61 datas, 10 bandas, 10 m); imagem SPOT 6-7 (1,5 m); rótulos de 18 culturas e ids de parcela; 5 folds oficiais separados espacialmente | Treino e avaliação dos Agents 01 e 02 | [IGNF/PASTIS-HD no Hugging Face](https://huggingface.co/datasets/IGNF/PASTIS-HD), etalab-2.0 |
| **RPG 2019**: polígono, cultura e área declarados à Política Agrícola Comum | Rótulos de borda do Agent 02 e objeto da auditoria de área | [IGN Géoplateforme](https://data.geopf.fr/telechargement/resource/RPG), Licence Ouverte |

Foram usados 150 recortes com pelo menos 5% de trigo (30 por fold; ~3 GB baixados) e 10.168 polígonos do RPG. Os dados não são versionados e são regenerados pelos scripts (ver [README técnico](agririsk_v2_pastis/README.md)). Fontes acessadas em 20/11/2026.

### Arquitetura

```
Sentinel-2 (26 × 10 × 128 × 128) ──> Agent 01 · U-TAE-lite ─────────────┐
                                     (cultura, confirmação, NDVI)        ├─> Agent 03 · região + risco ─> Agent 04 · parecer (LLM)
SPOT 2 m (3 × 640 × 640) + RPG ────> Agent 02 · U-Net multitarefa ───────┘        ↑
                                     (extensão, bordas, auditoria de área)   benchmark: trigos da mesma região
```

### Agentes e aderência ao curso

Os módulos citados são os da [página do curso](https://ica.ele.puc-rio.br/cursos/computer-vision-master/): (1) fundamentos e programação em Python, (2) aprimoramento de qualidade de imagens, (3) classificação e detecção avançadas, (4) síntese de imagens e modelos generativos.

#### Agent 01 · segmentação espaço-temporal de culturas (U-TAE-lite)

- **O que faz:** segmenta 18 culturas pixel a pixel a partir de 26 imagens Sentinel-2 e decide, por parcela, se a cultura declarada (trigo) é confirmada e desde quando.
- **Técnica:** encoder convolucional compartilhado entre as datas; atenção temporal multi-cabeça por pixel (L-TAE) com codificação senoidal da data; decoder U-Net em que os mesmos pesos de atenção agregam as conexões de cada escala; 0,56 M de parâmetros. Treino com AdamW e OneCycle, 80 épocas, entropia cruzada ponderada e máscara de datas futuras em 50% dos lotes (confirmação antecipada). Treinado numa GPU T4 (Google Colab) em 12 min.
- **Resultados (teste, fold 5):** F1 do trigo por parcela de 0,892 e AUC de 0,987, contra 0,849 e 0,985 do baseline Random Forest. IoU do trigo por pixel de 0,77 e mIoU de 0,41 nas 18 culturas. Até 13/abr, 75% dos trigos confirmados com 90% de precisão. Os erros concentram-se em outros cereais de inverno (triticale, trigo duro, cevada), de fenologia quase idêntica.
- **Aderência ao curso:**
    - **Módulo 3:** classificação de imagens com deep learning e CNNs; análise de conteúdo semântico na forma de segmentação semântica densa; avaliação com IoU, mIoU, F1 e AUC.
    - **Módulos 1 e 2:** manipulação de imagens multiespectrais, máscara de nuvens, composição temporal mediana, interpolação e normalização; índice espectral NDVI.
    - **Além do programa:** atenção temporal (o mecanismo dos transformers) aplicada a uma sequência de imagens, e interpretabilidade pela análise dos pesos de atenção.

#### Agent 02 · delimitação de talhões e auditoria de área (U-Net multitarefa)

- **O que faz:** segmenta a extensão cultivada e as bordas entre talhões na imagem SPOT e verifica se o polígono declarado engloba mais do que um talhão, o que indica área declarada a maior.
- **Técnica:** U-Net com encoder MobileNetV2 pré-treinado no ImageNet (transfer learning) e duas saídas sigmoid (extensão e borda), no estilo do ResUNet-a. Perda BCE + Dice com `pos_weight` para a classe rara (borda) e aumentação geométrica e fotométrica; 40 épocas numa T4, em 5 min. Instâncias por watershed sobre a probabilidade de borda. A auditoria compara a faixa externa do polígono com o núcleo: três variantes foram testadas e a escolha foi feita na validação.
- **Resultados (teste):** IoU de 0,871 na extensão; F1 de borda de 0,760 com tolerância de 4 m. Detecta 86,5% das áreas infladas em 30% e 66,9% das infladas em 15%, com 5,9% de alarmes falsos; a variante ingênua detectava só 5,8%. Só 48% das parcelas são separadas como instâncias próprias, porque parcelas vizinhas sem borda visível se fundem.
- **Aderência ao curso:**
    - **Módulo 3:** CNN para segmentação; transfer learning; detecção e delimitação de objetos (talhões) e métrica de objetos (IoU de instância).
    - **Módulos 1 e 2:** morfologia matemática (erosão, abertura), componentes conexos, transformada de distância, watershed e índice de vegetação a partir de RGB (VARI) para manchas de baixo vigor; georreferenciamento por transformação afim entre sistemas de coordenadas (Lambert-93 → pixel) para rasterizar os polígonos.

#### Agent 03 · contexto regional e score de risco

- **O que faz:** compara cada parcela com os trigos da mesma região (percentis do vigor acumulado, do NDVI de pico e da data do pico, calculados nos folds de treino) e combina declaração, área, vigor, ciclo e manchas num score auditável. O limiar de declaração é escolhido na validação com no máximo 3% de alarmes em trigos reais: detecta 87,7% das culturas falsas, com 2,5 alarmes a cada 100 trigos.
- **Aderência ao curso:** etapa de **integração**, em que as saídas dos modelos de visão viram decisão. Usa processamento de sinais derivados de imagem (suavização por mediana móvel da série de NDVI e métricas fenológicas por meia-amplitude), mas não é, em si, um modelo de visão computacional.

#### Agent 04 · parecer com LLM

- **O que faz:** gera um parecer em linguagem natural com saída estruturada (JSON Schema estrito), evidências restritas a um `enum` com os campos que existem na entrada e validação pós-geração. Foram comparados três modelos (gpt-4o-mini, gpt-4.1-mini e gpt-5.4-mini) em 60 parcelas; o gpt-5.4-mini foi escolhido por análise de erros.
- **Aderência ao curso:** **camada de aplicação**, fora do escopo de visão computacional. Mostra como comunicar a saída dos modelos de forma rastreável a um usuário não técnico.

> **Módulo 4 (síntese de imagens e modelos generativos)** não é coberto. Uma extensão natural seria a super-resolução da série Sentinel-2 guiada pela imagem SPOT.

### Resultados principais

| Tarefa | Métrica (teste, fold 5) | Valor |
|---|---|---|
| Identificar trigo por parcela (Agent 01) | F1 / AUC | **0,892 / 0,987** (Random Forest: 0,849 / 0,985) |
| Confirmação antecipada (Agent 01) | Trigos confirmados até 13/abr (precisão) | **75%** (90%) |
| Delimitação (Agent 02) | IoU da extensão / F1 de borda (4 m) | **0,871 / 0,760** |
| Auditoria de área (Agent 02) | Detecção de área +30% / alarme falso | **86,5% / 5,9%** |
| Conferência de cultura (Agent 03) | Culturas falsas detectadas / alarmes por 100 trigos | **87,7% / 2,5** |
| Parecer (Agent 04) | Pareceres aprovados na validação (60 parcelas) | 98% (gpt-5.4-mini) |

### Reprodução

Todo treino roda numa GPU T4 do Google Colab, e a avaliação numa VM de CPU do Colab, ambas via [google-colab-cli](https://github.com/googlecolab/google-colab-cli). Os passos, a configuração e os scripts estão no [README técnico](agririsk_v2_pastis/README.md).

### Limitações

- **Instâncias:** o rótulo administrativo (parcela declarada) nem sempre corresponde a uma fronteira visível na imagem.
- **Fraudes:** são simuladas (outra cultura declarada como trigo; polígono ampliado em +15% e +30%).
- **Abrangência:** uma safra (2018-19), uma região geográfica (França) e 150 dos 2.433 recortes.
- **Vigor:** não foi validado contra produtividade.
- **Parcelas pequenas:** abaixo de ~1,5 ha a auditoria de área não separa a faixa externa do núcleo; a parcela é marcada como não auditável (6 das 60 parcelas do teste ponta a ponta).

---

Matrícula: [preencher]

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós-Graduação *Visão Computacional Master* (VC Master)
