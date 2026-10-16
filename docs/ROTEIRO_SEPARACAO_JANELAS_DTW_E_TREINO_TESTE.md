# Especificação Técnica: Separação Física de Janelas (120s), Divisão Treino/Teste (1.700 / 534), Features Derivadas e Matriz DTW (1700x1700)

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** IFSULDEMINAS - Campus Poços de Caldas  
**Projeto:** Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos  

---

## 1. Objetivo e Motivação da Refatoração

Após revisão metodológica com o orientador, o pipeline de processamento de janelas comportamentais e treinamento não supervisionado foi evoluído para atender a quatro requisitos científicos fundamentais:

1. **Isolamento Físico por Janela Operacional:** Exportar cada uma das **2.234 janelas de condução ativa** ($120\text{ segundos}$, $\bar{v} \ge 8{,}0\text{ km/h}$ e $\ge 30$ amostras em $1\text{ Hz}$) em arquivos individuais nos formatos **`.csv`** e **`.parquet`**, permitindo auditoria direta e simulação de leitura individual em tempo real.
2. **Quebra de Dependência Temporal (*Shuffle*) e Divisão Estrita Treino/Teste:** Embaralhar de forma reprodutível (`random_state=42`) os 2.234 arquivos e separá-los fisicamente em:
   * **Conjunto de Treino (`data/windows/train/`):** **1.700 arquivos** ($\approx 76{,}1\%$) utilizados exclusivamente para o ajuste (*fit*) do `StandardScaler`, do `K-Means` ($k=3$), do `PCA` ($2\text{D}$) e da matriz de similaridade temporal **DTW**.
   * **Conjunto de Teste (`data/windows/test/`):** **534 arquivos** ($\approx 23{,}9\%$, totalizando $1700 + 534 = 2234$) isolados em pasta separada para avaliação *out-of-sample* de generalização do modelo e simulação de inferência em tempo real com a camada de **LLM**.
3. **Extração de Features Derivadas Enriquecidas por Janela:** Calcular variáveis espaciais e cinemáticas derivadas de cada arquivo de 120s (distância total percorrida, distância geodésica em linha reta via **Fórmula de Haversine**, índice de **tortuosidade/sinuosidade**, desaceleração média de frenagem, taxa de trancos por minuto e proporção de paradas), mantendo as **6 variáveis cinemáticas principais** como entrada pura do K-Means (**Opção A**).
4. **Análise de Similaridade Temporal via Matriz DTW ($1700 \times 1700$) sobre a Velocidade e Mapa de Calor (*Heatmap*):** Construir a matriz simétrica de distâncias *Dynamic Time Warping* (DTW) de dimensão **$1700 \times 1700$** exclusivamente sobre a série temporal de velocidade (`speed_kmh`) dos 1.700 arquivos de treino ($\frac{1700 \times 1699}{2} = 1.444.150$ pares únicos) e gerar o respectivo gráfico de calor (*heatmap*).

---

## 2. Arquitetura do Fluxo de Dados (7 Etapas)

```
[Telemetria Regularizada 1 Hz (ev_trips_1hz.parquet)]
        │
        ▼
[Etapa 1: Recorte das 2.234 Janelas Ativas de 120s (v_media >= 8.0 km/h)]
        │
        ▼
[Etapa 2: Embaralhamento Reprodutível (Shuffle com semente fixa random_state=42)]
        │
        ▼
[Etapa 3: Divisão Física em Pastas (.csv e .parquet)]
   ├── data/windows/train/csv/ & parquet/  --> 1.700 arquivos de Treino (76,1%)
   └── data/windows/test/csv/  & parquet/  -->   534 arquivos de Teste  (23,9%)
        │
        ├──────────────────────────────────────────────────────┐
        ▼                                                      ▼
[Etapa 4: Extração de Features Derivadas por Arquivo]   [Etapa 5: Matriz DTW 1700x1700]
   • 6 Features do K-Means (Opção A)                       • Série temporal: speed_kmh (1.700 treinos)
   • + Distância Total, Haversine, Tortuosidade            • 1.444.150 pares únicos via Numba JIT
   • + Jerk/min, Idle Ratio, Desaceleração Média           • Heatmap 1700x1700 (Ordem Shuffle e por Cluster)
        │                                                      ▲
        ▼                                                      │
[Etapa 6: Treinamento Estrito nos 1.700 Arquivos (train/)] ────┘
   • Fit exclusivo de StandardScaler + K-Means (k=3) + PCA (2D)
        │
        ▼
[Etapa 7: Validação Out-of-Sample e Predição nos 534 Arquivos (test/)]
   • Transform + Predict sem reajuste de centroides
   • Validação de Silhueta, Distribuição de Perfis e Potência (kW)
   • Base de 534 janelas inéditas pronta para inferência em tempo real + LLM
```

---

## 3. Detalhamento Matemático das Novas Componentes

### 3.1. Features Derivadas Espaciais e Cinemáticas por Janela (120s)

Para cada arquivo individual de janela contendo $N \le 120$ pontos em $1\text{ Hz}$:

1. **Distância Real Percorrida ($D_{\text{total}}$, em metros):**
   $$D_{\text{total}} = \sum_{t=1}^{N} \frac{v_t}{3{,}6} \cdot \Delta t$$
   onde $v_t$ é a velocidade instantânea em $\text{km/h}$ e $\Delta t = 1\text{ s}$.

2. **Distância em Linha Reta via Fórmula de Haversine ($D_{\text{reta}}$, em metros):**
   Dadas as coordenadas GPS inicial $(\phi_1, \lambda_1)$ e final $(\phi_N, \lambda_N)$ da janela convertidas para radianos e o raio médio da Terra $R = 6.371.000\text{ m}$:
   $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_N)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
   $$D_{\text{reta}} = 2 R \cdot \arctan2\left(\sqrt{a}, \sqrt{1 - a}\right)$$

3. **Índice de Tortuosidade / Sinuosidade da Rota ($I_{\text{tort}}$):**
   $$I_{\text{tort}} = \frac{D_{\text{total}}}{\max(D_{\text{reta}}, 1{,}0)}$$
   Valores próximos de $1{,}0$ indicam deslocamento retilíneo (ex.: avenida arterial ou rodovia expressa), enquanto valores elevados ($> 1{,}3$) indicam trajetos urbanos sinuosos, conversões sucessivas em quarteirões ou rotatórias.

4. **Indicadores Dinâmicos Complementares:**
   * **`mean_neg_accel_ms2`:** Média das desacelerações ($a_t < -0{,}1\text{ m/s}^2$), quantificando a severidade média das frenagens na janela.
   * **`high_jerk_rate_min`:** Taxa de eventos de tranco severo ($|j_t| \ge 3{,}0\text{ m/s}^3$) normalizada por minuto.
   * **`idle_ratio`:** Proporção do tempo da janela em que o veículo permaneceu com velocidade quase nula ($v_t < 2{,}0\text{ km/h}$), capturando retenções semafóricas.

> **Nota de Modelagem (Opção A):** O treinamento do K-Means mantém exclusivamente as **6 features cinemáticas validadas** (`mean_speed_kmh`, `std_speed_kmh`, `mean_pos_accel_ms2`, `max_pos_accel_ms2`, `hard_braking_rate_min`, `rapid_accel_rate_min`) para preservar a pureza da classificação de estilo do motorista sem confundir topografia/geometria viária com agressividade humana. As novas features derivadas são armazenadas no dataset rotulado para análise contextual e envio à camada de LLM.

---

### 3.2. Matriz DTW ($1700 \times 1700$) sobre a Velocidade (`speed_kmh`) e Heatmap

O **Dynamic Time Warping (DTW)** mede a distância de alinhamento elástico não-linear entre duas séries temporais de velocidade $X = (x_1, x_2, \dots, x_n)$ e $Y = (y_1, y_2, \dots, y_m)$ pertencentes ao conjunto de treino ($1.700$ janelas):

$$D(i, j) = |x_i - y_j| + \min \begin{cases} D(i-1, j) \\ D(i, j-1) \\ D(i-1, j-1) \end{cases}$$

com distância final $\text{DTW}(X, Y) = D(n, m)$ (ou versão euclidiana acumulada $\sqrt{\sum (x_i - y_j)^2}$).

* **Dimensão da Matriz:** $1700 \times 1700$ (simétrica, com diagonal principal $D_{ii} = 0$, exigindo o cálculo de $\frac{1700 \times 1699}{2} = 1.444.150$ pares de séries temporais de até 120 pontos).
* **Aceleração Computacional:** Implementada com compilação *Just-In-Time* paralelizada (`@njit(parallel=True, fastmath=True)` via `numba`), permitindo calcular os $1{,}44\text{ milhão}$ de alinhamentos em poucos segundos.
* **Visualização em Mapa de Calor (*Heatmap*):**
  O arquivo `outputs/figures/07_dtw_speed_heatmap_1700x1700.png` (e piloto `07_dtw_speed_heatmap_pilot_100x100.png`) apresenta a matriz de distâncias DTW:
  1. **Na ordem embaralhada do conjunto de treino (`train_0000` a `train_1699`):** Demonstra visualmente que o embaralhamento (*shuffle*) eliminou correlações espúrias entre janelas adjacentes da mesma viagem.
  2. **Reordenada pelos perfis de condução:** Evidencia a formação de blocos estruturados na diagonal principal.

---

### 3.3. Agrupamento por Detecção de Comunidades: O Algoritmo de Louvain

A partir da matriz de distâncias DTW $D$, aplica-se o algoritmo de **Louvain** (Blondel et al., 2008) para identificar as comunidades naturais de condução baseadas na topologia de rede:

1. **Construção do Grafo de Afinidade ($W_{ij}$):**
   Converte-se a matriz de distâncias em similaridade via kernel exponencial:
   $$W_{ij} = \exp\left(-\frac{D_{ij}}{\sigma}\right) \quad \text{para } i \neq j, \quad W_{ii} = 0$$
   onde $\sigma = \text{mediana}(D)$, associado a um grafo esparso de vizinhos mais próximos ($k\text{-NN}$) para remover ruído e focar nas conexões de alta similaridade temporal.
2. **Otimização da Modularidade ($Q$):**
   O algoritmo de Louvain busca a partição de janelas que maximiza a densidade de conexões internas nas comunidades em relação a uma rede aleatória nula, **sem necessidade de fixar previamente o número de clusters $k$**.
3. **Novo Heatmap com Divisórias das Comunidades:**
   Reordenando as janelas pelas comunidades descobertas pelo Louvain, gera-se a figura `outputs/figures/08_dtw_louvain_heatmap_pilot_100x100.png` (e futura versão 1700x1700), exibindo blocos escuros na diagonal principal delimitados por linhas estruturais que comprovam visualmente a separação dos estilos de condução.

---

## 4. Estrutura de Diretórios e Artefatos Gerados

```
data/
├── windows/                                     # (Ignorado no Git via .gitignore)
│   ├── train/
│   │   ├── csv/                                 # 1.700 arquivos .csv de treino (120s cada)
│   │   └── parquet/                             # 1.700 arquivos .parquet de treino (120s cada)
│   └── test/
│       ├── csv/                                 # 534 arquivos .csv de teste (120s cada)
│       └── parquet/                             # 534 arquivos .parquet de teste (120s cada)
└── processed/
    ├── window_split_manifest.csv                # Manifesto rastreável das 2.234 janelas (train vs test)
    ├── driving_windows_train_features.parquet   # 1.700 janelas de treino com features extraídas
    ├── driving_windows_test_features.parquet    # 534 janelas de teste com features extraídas
    ├── driving_windows_train_labeled.parquet    # 1.700 janelas de treino com features + rótulos K-Means
    ├── driving_windows_test_labeled.parquet     # 534 janelas de teste com features + predição K-Means
    ├── dtw_speed_matrix_pilot_100x100.npy       # Matriz DTW piloto 100x100 (NumPy)
    ├── dtw_speed_matrix_pilot_100x100.parquet   # Matriz DTW piloto 100x100 (Parquet estruturado)
    ├── dtw_speed_matrix_train_1700x1700.npy     # Matriz DTW final 1700x1700 (NumPy)
    └── dtw_louvain_clusters_pilot_100x100.csv   # Mapeamento de comunidades Louvain (Piloto)

outputs/
└── figures/
    ├── 07_dtw_speed_heatmap_pilot_100x100.png   # Heatmap piloto 100x100 (Ordem Shuffle)
    ├── 07_dtw_speed_heatmap_1700x1700.png       # Heatmap final 1700x1700 (Ordem Shuffle)
    └── 08_dtw_louvain_heatmap_pilot_100x100.png # Heatmap reordenado pelas comunidades de Louvain
```
