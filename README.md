# 🫀 CardioIA — Fase 1: Batimentos de Dados

> **Projeto Acadêmico** | FIAP — Curso de Inteligência Artificial  
> **Fase 1 — Batimentos de Dados** | Ciência de Dados aplicada à Cardiologia

---

## 📋 Sobre o Projeto

O **CardioIA** é uma plataforma digital inteligente que simula o ecossistema de uma cardiologia moderna, integrando dados clínicos, modelos de Machine Learning, Visão Computacional, IoT e agentes inteligentes para triagem, diagnóstico, monitoramento e previsão de doenças cardíacas.

Nesta **Fase 1 — Batimentos de Dados**, o objetivo é levantar, organizar e documentar três tipos de dados fundamentais que alimentarão os módulos inteligentes do CardioIA nas fases seguintes:

| # | Tipo de Dado | Volume | Finalidade |
|---|---|---|---|
| 1 | 📊 Numérico (IoT) | 500 registros de pacientes | Machine Learning / Predição |
| 2 | 📝 Textual (NLP) | 2 textos médicos (.txt) | Processamento de Linguagem Natural |
| 3 | 🖼️ Visual (VC) | 102 imagens de ECG (.png) | Visão Computacional |

---

## 🗂️ Estrutura do Repositório

```
cardioIA/
├── README.md                               # Este arquivo
├── heart_disease_dataset.csv               # Dataset numérico (500 pacientes)
├── docs/
│   ├── texto1_doencas_cardiovasculares.txt # Texto médico 1 — Epidemiologia das DCV
│   └── texto2_ecg_e_monitoramento.txt      # Texto médico 2 — ECG e Monitoramento
└── assets/
    └── images/
        ├── ecg_normal_001.png              # ECG: Ritmo Sinusal Normal
        ├── ecg_stemi_001.png               # ECG: IAMCSST
        ├── ecg_ischemia_001.png            # ECG: Isquemia
        ├── ecg_afib_001.png                # ECG: Fibrilação Atrial
        └── ...                             # (102 imagens no total)
```

---

## ☁️ Links para os Dados Completos

> ⚠️ **Importante:** Os links abaixo devem ser substituídos pelos seus links reais no Google Drive ou OneDrive após o upload dos arquivos.

| Tipo | Descrição | Link |
|------|-----------|------|
| 📊 Dataset Numérico | `heart_disease_dataset.csv` (500 pacientes) | [🔗 Abrir no Drive](#) |
| 📝 Textos Médicos | Pasta `docs/` com 2 arquivos .txt | [🔗 Abrir no Drive](#) |
| 🖼️ Imagens de ECG | Pasta `assets/images/` com 102 imagens .png | [🔗 Abrir no Drive](#) |
| 📦 Projeto Completo | Pasta raiz com todos os dados | [🔗 Abrir no Drive](#) |

---

## 📊 Parte 1 — Dados Numéricos (IoT)

### Origem dos Dados

O dataset `heart_disease_dataset.csv` contém **500 registros simulados** de pacientes cardíacos, gerados computacionalmente com base nas distribuições estatísticas e correlações clínicas descritas nos seguintes datasets públicos reais:

- **UCI Heart Disease Dataset** (Cleveland, Hungarian, VA Long Beach, Switzerland) — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Framingham Heart Study** — dados de referência para distribuições epidemiológicas
- **Diretrizes da Sociedade Brasileira de Cardiologia** — para limites e referências clínicas

Os dados foram simulados com aleatoriedade controlada (`random.seed(42)`) para garantir reproducibilidade, preservando as correlações clínicas esperadas entre as variáveis (ex.: maior prevalência de doença cardíaca em pacientes mais velhos, do sexo masculino e com múltiplos fatores de risco).

### Variáveis do Dataset

| Variável | Tipo | Descrição | Relevância Clínica |
|---|---|---|---|
| `id_paciente` | string | Identificador único | Controle |
| `idade` | int | Idade em anos | ⭐⭐⭐ Fator de risco não modificável; risco aumenta após 45a (H) e 55a (M) |
| `sexo` | string (M/F) | Sexo biológico | ⭐⭐⭐ Sexo masculino associado a maior risco nas faixas jovens |
| `tipo_dor_toracica` | categórica | Angina típica, atípica, sem angina, assintomático | ⭐⭐⭐ Principal sintoma guia de isquemia miocárdica |
| `pressao_arterial_sistolica` | int (mmHg) | Pressão arterial sistólica | ⭐⭐⭐ HAS é o principal fator de risco modificável para DCV |
| `pressao_arterial_diastolica` | int (mmHg) | Pressão arterial diastólica | ⭐⭐ Associada a risco de IC e AVC |
| `colesterol_mg_dl` | float | Colesterol total em mg/dL | ⭐⭐⭐ Dislipidemia é fator central da aterosclerose |
| `glicemia_jejum_120` | binário (0/1) | Glicemia em jejum > 120 mg/dL | ⭐⭐ Indicador de diabetes/pré-diabetes |
| `resultado_ecg_repouso` | categórica | Normal, anormalidade ST, hipertrofia VE | ⭐⭐⭐ Reflete alterações estruturais e isquêmicas |
| `fc_maxima_atingida` | int (bpm) | Frequência cardíaca máxima no esforço | ⭐⭐⭐ FC máx reduzida sugere comprometimento funcional |
| `fc_repouso` | int (bpm) | Frequência cardíaca em repouso | ⭐⭐ Taquicardia em repouso pode indicar IC ou disfunção autonômica |
| `angina_induzida_exercicio` | binário (0/1) | Angina precipitada pelo exercício | ⭐⭐⭐ Forte indicador de isquemia miocárdica induzida |
| `depressao_st_exercicio` | float (mm) | Depressão do segmento ST no exercício | ⭐⭐⭐ Critério diagnóstico de isquemia no teste ergométrico |
| `inclinacao_st` | categórica | Ascendente, plana, descendente | ⭐⭐ Padrão descendente é o mais sugestivo de isquemia |
| `num_vasos_coloridos_fluoroscopia` | int (0-3) | Nº de vasos com estenose na cinecoronariografia | ⭐⭐⭐ Reflete extensão da doença arterial coronariana |
| `talassemia` | categórica | Normal, defeito fixo, defeito reversível | ⭐⭐ Detectada na cintilografia miocárdica |
| `imc` | float (kg/m²) | Índice de Massa Corporal | ⭐⭐ Obesidade (IMC > 30) é fator de risco independente |
| `fumante` | binário (0/1) | Tabagismo atual | ⭐⭐⭐ Causa direta de aterosclerose e trombose |
| `diabetes` | binário (0/1) | Diagnóstico de diabetes mellitus | ⭐⭐⭐ Multiplica o risco cardiovascular em 2-4x |
| `historico_familiar_cardiaco` | binário (0/1) | Familiar de 1º grau com DCV prematura | ⭐⭐ Fator genético independente |
| `doenca_cardiaca` | binário (0/1) | **Variável alvo (target)** | ⭐⭐⭐ Label para classificação supervisionada |

### Variáveis Mais Relevantes para IA

As variáveis mais importantes para um modelo de Machine Learning de predição de doença cardíaca são:

1. **`tipo_dor_toracica`** — É o sintoma com maior valor preditivo positivo isolado para doença arterial coronariana.
2. **`depressao_st_exercicio`** — Critério objetivo e quantitativo de isquemia, amplamente validado clinicamente.
3. **`num_vasos_coloridos_fluoroscopia`** — Reflete diretamente a extensão anatômica da doença coronariana.
4. **`idade` + `sexo`** — Interação fundamental; o risco se distribui de forma diferente entre sexos conforme a faixa etária.
5. **`colesterol_mg_dl`** — Biomarcador clássico do processo aterosclerótico.
6. **`fc_maxima_atingida`** — A incapacidade de atingir a FC máxima prevista é marcador de mau prognóstico.

### Considerações sobre Governança e Viés

- **Dados simulados**: Por serem sintéticos, não há risco de exposição de dados pessoais sensíveis, em conformidade com a LGPD (Lei Geral de Proteção de Dados — Lei nº 13.709/2018) e os princípios de Privacy by Design.
- **Viés de representatividade**: O dataset simula distribuições de estudos predominantemente realizados em populações norte-americanas e europeias. Em uma aplicação real no Brasil, seria necessário validar e recalibrar o modelo com dados da população brasileira, que apresenta perfil epidemiológico e genético distinto.
- **Viés de gênero**: Estudos históricos subestimaram a presença e manifestação de DCV em mulheres. O dataset reflete essa distribuição conhecida, mas modelos treinados nele devem ser avaliados quanto a desempenho diferencial por sexo.
- **Balanceamento de classes**: 35,8% de casos positivos (doença cardíaca) vs. 64,2% negativos — leve desbalanceamento que pode exigir técnicas como SMOTE ou ajuste de pesos de classe no treinamento.

---

## 📝 Parte 2 — Dados Textuais (NLP)

### Textos Disponíveis

| Arquivo | Tema | Tamanho Aprox. |
|---|---|---|
| `docs/texto1_doencas_cardiovasculares.txt` | Panorama epidemiológico e clínico das DCV | ~6.500 palavras |
| `docs/texto2_ecg_e_monitoramento_cardiaco.txt` | ECG, monitoramento cardíaco e IA | ~5.800 palavras |

Os textos foram elaborados com linguagem técnico-científica, cobrindo epidemiologia, fisiopatologia, diagnóstico, tratamento e aplicações de IA na cardiologia, com base em diretrizes da Sociedade Brasileira de Cardiologia (SBC), dados do Ministério da Saúde e literatura científica indexada.

### Aplicações de NLP nos Textos Médicos

Os textos coletados podem ser explorados por algoritmos de NLP de diversas formas, todas altamente relevantes para o projeto CardioIA:

#### 1. Extração de Entidades Nomeadas (NER — Named Entity Recognition)
Identificação automática de entidades como nomes de doenças (infarto agudo do miocárdio, fibrilação atrial), medicamentos (ácido acetilsalicílico, estatinas), exames (ECG, ecocardiograma), sintomas (dispneia, dor torácica) e valores clínicos (pressão arterial ≥ 140 mmHg). Essa extração estrutura o conhecimento médico não-estruturado dos textos para alimentar sistemas de suporte à decisão clínica.

#### 2. Classificação de Tópicos (Topic Modeling)
Algoritmos como LDA (Latent Dirichlet Allocation) ou BERTopic podem identificar automaticamente os temas predominantes em cada texto — como fatores de risco, diagnóstico, tratamento e epidemiologia — sem necessidade de anotação manual. Útil para organizar grandes volumes de literatura médica.

#### 3. Análise de Sentimentos e Confiança Clínica
Modelos de análise de sentimento adaptados ao domínio médico podem identificar o grau de certeza/incerteza nas afirmações clínicas dos textos (ex.: "está associado a maior risco" vs. "pode estar associado"). Essa nuance é importante em sistemas de apoio à decisão.

#### 4. Extração de Relações Semânticas
Identificação de relações entre entidades, como "tabagismo CAUSA aterosclerose" ou "metformina TRATA diabetes". Essas relações podem alimentar grafos de conhecimento médico (Knowledge Graphs) usados por agentes inteligentes do CardioIA.

#### 5. Geração de Resumos Automáticos (Summarization)
Modelos de linguagem como BERT, GPT ou T5 podem gerar resumos concisos dos textos, úteis para criação de boletins informativos automatizados, relatórios para pacientes em linguagem acessível e síntese de literatura para equipes clínicas.

#### 6. Resposta a Perguntas (Question Answering)
Um sistema de QA treinado nestes textos pode responder a perguntas como "Quais são os sintomas da angina instável?" ou "Qual é o tratamento de primeira linha para hipertensão?" diretamente a partir do conteúdo dos documentos.

### Justificativa para IA em Saúde

O processamento automatizado de textos médicos é estratégico porque a maior parte do conhecimento clínico ainda está armazenada em formato não estruturado: prontuários eletrônicos, laudos, artigos científicos e diretrizes. NLP permite transformar esse conhecimento em dados estruturados, acessíveis a modelos de ML, sistemas de alertas e interfaces conversacionais para pacientes e profissionais de saúde.

---

## 🖼️ Parte 3 — Dados Visuais (Visão Computacional)

### Descrição das Imagens

Foram reunidas **102 imagens sintéticas de ECG** (Eletrocardiograma) em formato `.png`, organizadas por tipo/patologia:

| Tipo de ECG | Quantidade | Descrição Clínica |
|---|---|---|
| Normal (Ritmo Sinusal) | 25 | ECG com morfologia e intervalos dentro dos padrões normais |
| IAMCSST (Supradesnivelamento ST) | 11 | Padrão de infarto agudo com oclusão coronariana |
| Isquemia (Inversão Onda T) | 11 | Alterações sugestivas de isquemia subepicárdica |
| Infradesnivelamento ST | 11 | Sugestivo de isquemia subendocárdica ou sobrecarga |
| Hipertrofia Ventricular Esquerda | 11 | Padrão de HVE com critérios de voltagem aumentados |
| Fibrilação Atrial | 11 | Ausência de onda P, resposta ventricular irregular |
| Taquicardia Ventricular | 11 | QRS alargado, alta frequência, morfologia aberrante |
| Bloqueio de Ramo Esquerdo | 11 | QRS alargado com morfologia característica de BRE |
| **Total** | **102** | |

As imagens foram geradas sinteticamente com bibliotecas Python (Pillow), respeitando as morfologias clínicas de cada condição cardíaca, com grade eletrocardiográfica padrão e variações realistas entre exemplares da mesma categoria.

### Aplicações de Visão Computacional

#### 1. Classificação de Padrões (Image Classification)
Redes neurais convolucionais (CNN) como ResNet, EfficientNet ou Vision Transformers (ViT) podem ser treinadas para classificar automaticamente o tipo de ECG — normal, IAMCSST, FA, etc. — atingindo desempenho comparável ao de cardiologistas em estudos publicados em revistas como Nature Medicine.

#### 2. Detecção de Objetos e Segmentação de Ondas (Object Detection)
Modelos como YOLO ou Mask R-CNN podem ser adaptados para detectar e segmentar as ondas individuais do ECG (P, QRS, T, segmento ST), permitindo medição automática de intervalos e amplitudes. Isso é equivalente ao trabalho manual de um técnico treinado em ECG.

#### 3. Detecção de Anomalias (Anomaly Detection)
Autoencoders e modelos generativos (GANs, VAEs) treinados em ECGs normais podem identificar traçados anômalos com base na diferença de reconstrução, sendo úteis para alertas precoces em monitoramento contínuo.

#### 4. Reconhecimento de Padrões Temporais (Temporal Pattern Recognition)
O ECG é um sinal temporal. Redes LSTM e Transformer aplicadas a sequências de pixels ou sinais extraídos das imagens podem capturar padrões temporais de arritmias e alterações de repolarização.

#### 5. Aumento de Dados (Data Augmentation)
As imagens sintéticas podem ser augmentadas (rotação, escala, adição de ruído, variação de contraste) para aumentar artificialmente o volume de dados de treinamento, técnica essencial em domínios com dados escassos como a cardiologia.

### Justificativa para IA em Saúde

O ECG é realizado bilhões de vezes ao ano no mundo, mas a análise especializada é limitada pela disponibilidade de cardiologistas. Sistemas de Visão Computacional para interpretação automática de ECGs podem democratizar o acesso ao diagnóstico cardiológico de qualidade, especialmente em regiões remotas e países de baixa e média renda, onde a escassez de especialistas é crítica. A integração com dispositivos IoT de monitoramento (smartwatches, patches de ECG) potencializa ainda mais o impacto clínico dessas tecnologias.

---

## ⚖️ Governança de Dados e Considerações Éticas

### LGPD e Privacidade
- Todos os dados numéricos são **simulados** — não há dados reais de pacientes, eliminando riscos de privacidade e dispensando autorização de comitê de ética (CEP/CONEP) para esta fase.
- Os textos utilizados são de **domínio público ou uso educacional**.
- As imagens de ECG são **sintéticas**, geradas algoritmicamente.

### Viés Algorítmico
- O dataset numérico reflete vieses históricos da literatura cardiológica (sub-representação de mulheres, populações não-europeias).
- Modelos treinados neste dataset devem ser **validados externamente** em populações diversas antes de qualquer aplicação clínica.
- Em fases futuras, recomenda-se buscar datasets com maior diversidade demográfica, como o **MIMIC-IV** (PhysioNet) ou dados do **DATASUS**.

### Qualidade e Rastreabilidade
- As distribuições estatísticas do dataset foram baseadas em estudos publicados e referenciados.
- O código de geração dos dados está disponível no repositório, garantindo **reproducibilidade total**.
- A versão dos dados e do código estão documentadas para auditoria.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.x** — Geração de dados e imagens sintéticas
- **Pillow (PIL)** — Geração de imagens PNG de ECG
- **CSV** — Formato do dataset numérico
- **Markdown** — Documentação

---

## 👥 Equipe

| Nome | RM | Função |
|---|---|---|
| [Nome do Aluno 1] | RM000000 | Líder / Cientista de Dados |
| [Nome do Aluno 2] | RM000000 | Engenharia de Dados |
| [Nome do Aluno 3] | RM000000 | Documentação e Governança |

---

## 📚 Referências

1. **UCI ML Repository** — Heart Disease Dataset. [https://archive.ics.uci.edu/ml/datasets/heart+disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. **Sociedade Brasileira de Cardiologia (SBC)** — Diretrizes e Consensos. [https://www.cardiol.br](https://www.cardiol.br)
3. **Ministério da Saúde** — Doenças Cardiovasculares. [https://www.gov.br/saude](https://www.gov.br/saude)
4. **PhysioNet** — Base de dados de sinais fisiológicos. [https://physionet.org](https://physionet.org)
5. **SciELO Brasil** — Literatura científica em saúde. [https://www.scielo.br](https://www.scielo.br)
6. **BVS (Biblioteca Virtual em Saúde)** — [https://bvsalud.org](https://bvsalud.org)
7. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. *Nat Med*. 2019;25:44–56.
8. Hannun AY, et al. Cardiologist-level arrhythmia detection using deep neural networks. *Nat Med*. 2019;25:65–69.

---

> 🫀 **CardioIA** — FIAP | Curso de Inteligência Artificial | Fase 1 — Batimentos de Dados
