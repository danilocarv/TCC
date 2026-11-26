# Análise Comparativa da Fase 2: Regressão Linear (Baseline), XGBoost e MLP na Predição de Consumo Energético de Veículos Elétricos

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** IFSULDEMINAS – Campus Poços de Caldas  
**Base de Dados:** Vehicle Energy Dataset (VED) – 3 unidades do Nissan Leaf 2013 (`VehId 10, 455, 541`, bateria nominal de `24 kWh`)  
**Objetivo do Documento:** Registrar de forma completa o tratamento de integridade da base de dados da Fase 2 (com destaque para o filtro de lacunas de gravação OBD), a implementação do baseline e do modelo MLP frente ao XGBoost, os resultados consolidados nas `482` viagens válidas, as conclusões científicas e técnicas e as recomendações para uma eventual implementação futura em tempo real.

---

## 1. Base de Dados, Filtro de Integridade OBD e Protocolo Experimental

Após a revisão da versão preliminar do TCC pelo orientador, incluiu-se na Fase 2 um **baseline claro** (`Regressão Linear Múltipla`, além do piso de sanidade `DummyRegressor`) e um **terceiro algoritmo preditivo** (`MLP Regressor`) para avaliar de forma comparativa o ganho real de cada abordagem frente ao `XGBoost`. Paralelamente, uma auditoria fina de ponta a ponta sobre os dados brutos de telemetria revelou um critério crítico de qualidade de dados que se tornou um dos pontos centrais de saneamento da Fase 2.

### 1.1 Ponto Principal de Saneamento da Fase 2: Filtro de Integridade de Gravação OBD (`max_obd_gap_s ≤ 60 s`)

A base consolidada de veículos elétricos do VED possui **`504` viagens registradas** (`VehId 455`: `293`; `VehId 10`: `201`; `VehId 541`: `10`), somando `237.045` registros reamostrados a `1 Hz` (`data/processed/driving_features/ev_trips_1hz.parquet`).

Inicialmente, o critério de validade descartava apenas **`18` viagens curtas ou estacionárias** (`duração < 60 s` ou `distância < 0,5 km`), restando `486` viagens. Contudo, a inspeção das séries temporais individuais identificou **`4` viagens (todas do `VehId 455`) com lacunas contínuas de gravação OBD superiores a 10 minutos** dentro da mesma viagem:

| Veículo (`VehId`) | Viagem (`Trip`) | Maior lacuna contínua OBD (`max_obd_gap_s`) | Problema físico gerado nos alvos |
| :---: | :---: | :---: | :--- |
| `455` | `1682` | `1.318 s` (`22,0 min`) | Queda real de bateria durante o corte não é integrada em `energy_net_kwh` |
| `455` | `1795` | `916 s` (`15,3 min`) | Distorção severa entre energia integrada e queda de SOC |
| `455` | `1418` | `681 s` (`11,4 min`) | Desalinhamento entre `ΔSOC` (`SOC_inicial − SOC_final`) e integral de potência |
| `455` | `1509` | `651 s` (`10,9 min`) | Desalinhamento entre `ΔSOC` e integral de potência |

**Por que esse filtro é fisicamente indispensável?**
* Na extração de atributos (`src/trip_feature_extraction.py`), a energia líquida (`energy_net_kwh`) é calculada pela integração temporal $E = \int P(t)\,dt$ com limitação `dt.clip(1, 5)` segundos para evitar que falhas de gravação multipliquem a última potência lida por vários minutos.
* Já a variação do estado de carga (`delta_soc_percent`) é obtida pela diferença direta entre o início e o fim da viagem ($\Delta\text{SOC} = \text{SOC}_{\text{inicial}} - \text{SOC}_{\text{final}}$).
* Quando o gravador OBD interrompe a captura por 11 a 22 minutos e depois retoma a mesma viagem, a energia integrada "congela" durante o apagão, mas o `ΔSOC` contabiliza toda a descarga ocorrida no intervalo sem telemetria (gerando anomalias físicas como quedas próximas de `18%` de SOC associadas a menos de `1 kWh` integrado). Além de corromperem a relação física entre variáveis de entrada e `ΔSOC`, essas `4` viagens atuavam como *outliers* de gravação que degradavam o treinamento e a avaliação de todos os algoritmos.

Com a inclusão da regra **`max_obd_gap_s <= 60.0` segundos** no critério `is_valid_trip` de `src/trip_feature_extraction.py`:

$$\text{is\_valid\_trip} = (\text{duration\_s} \ge 60) \;\land\; (\text{distance\_km} \ge 0{,}5) \;\land\; (\text{max\_obd\_gap\_s} \le 60)$$

* **Viagens descartadas (`22` de `504`):** `18` curtas/estacionárias + `4` com falha de gravação OBD (`> 60 s`).
* **Base final saneada da Fase 2:** **`482` viagens válidas** (`VehId 455`: `283`; `VehId 10`: `189`; `VehId 541`: `10`), totalizando `2.290,97 km` percorridos (`0,51` a `21,50 km`, média `4,75 km`), duração média de `9,26 min` (`1,02` a `64,70 min`), energia líquida acumulada de `360,768 kWh` (`465,82 kWh` brutos menos `105,051 kWh` regenerados; eficiência média de regeneração de `25,75%`), `53,019 kWh` de consumo auxiliar de HVAC, ganho médio de elevação de `67,48 m` (máximo `270,0 m`) e temperatura externa média de `12,15 °C` (`−14,45 °C` a `35,51 °C`).

#### Impacto Direto do Filtro OBD (`486` viagens × `482` viagens)

A remoção das `4` viagens corrompidas produziu melhora expressiva e consistente nos três algoritmos, principalmente no alvo `ΔSOC`, além de eliminar todas as previsões negativas do `MLP` no teste:

| Modelo e Alvo | Antes (`486` viagens: `388`/`98`) | Depois do Filtro OBD (`482` viagens: `385`/`97`) | Ganho Observado |
| :--- | :---: | :---: | :--- |
| **Regressão Linear — `ΔSOC` (Teste R² / RMSE)** | `0,6906` / `2,0325 %` | **`0,8727` / `1,5314 %`** | `+18,21 p.p.` em R²; `−24,7%` no RMSE |
| **XGBoost — `ΔSOC` (Teste R² / RMSE)** | `0,7336` / `1,8859 %` | **`0,8479` / `1,6734 %`** | `+11,43 p.p.` em R²; `−11,3%` no RMSE |
| **MLP — `ΔSOC` (CV 5-Fold R²)** | `0,8523 ± 0,0580` | **`0,8706 ± 0,0182`** | `+1,83 p.p.` em R² e desvio entre *folds* `3,2×` menor |
| **MLP — `ΔSOC` (Teste R² / MAE / RMSE / Neg.)** | `0,8000` / `1,1020 %` / `1,6342 %` / `1 neg.` | **`0,8897` / `1,0642 %` / `1,4252 %` / `0 neg.`** | `+8,97 p.p.` em R²; `−12,8%` no RMSE; **zero negativos** |
| **MLP — Energia Líquida (Teste R² / Neg.)** | `0,9616` / `0 neg.` | **`0,9757` / `0 neg.`** | `+1,41 p.p.` em R² (`MAE = 0,0790 kWh`) |

### 1.2 Protocolo Experimental Unificado (Sem *Data Leakage*)

Para garantir comparação estritamente justa entre todos os algoritmos:

* **Base consolidada:** `data/processed/trips/ev_trips_complete_dataset.parquet` (`482` viagens válidas).
* **Por que modelagem em nível de viagem (*trip-level*) e não segundo a segundo (`1 Hz`)?**
  * Todos os `237.045` registros de `1 Hz` foram usados na etapa de extração (`src/trip_feature_extraction.py`) para integrar distância, acelerações, variabilidade de velocidade, relevo SRTM, energia de HVAC e balanço energético com frenagem regenerativa.
  * Consolidar em 1 linha por viagem completa é o padrão adequado para modelos tabulares (`Regressão Linear`, `XGBoost` e `MLP`), pois: (i) viagens têm durações muito diferentes e esses algoritmos exigem entrada de dimensão fixa (`12` atributos); (ii) o alvo `ΔSOC` varia em degraus discretos no painel (sendo `0%` na maioria dos segundos individuais); (iii) impede vazamento de segundos vizinhos entre treino e teste; e (iv) reflete o uso prático em estimativa de autonomia de rota.
* **Divisão Treino / Teste (*Holdout*) e Embaralhamento:** `train_test_split(..., test_size=0.20, random_state=42, stratify=pd.qcut(distance_km, q=5))` (`shuffle=True` padrão), separando deterministicamente **`385` viagens de treino (80%)** e **`97` viagens inéditas de teste (20%)** com a mesma proporção de trajetos curtos e longos.
* **Validação Cruzada Interna:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` estratificada em 5 quantis de distância dentro das `385` viagens de treino. No desenvolvimento do MLP, o conjunto de teste (*holdout*, `97` viagens) permaneceu **completamente isolado** durante todo o diagnóstico e seleção de hiperparâmetros, sendo avaliado apenas na etapa final.
* **12 Variáveis de Entrada (`FEATURE_COLS`):**
  1. *Cinemática & Operação (6):* `distance_km`, `duration_s`, `mean_speed_kmh`, `std_speed_kmh`, `mean_pos_accel_ms2`, `mean_neg_accel_ms2`.
  2. *Topografia — NASA SRTM 30 m (`data/raw/dem/N42W084.hgt`) (3):* `elevation_gain_m`, `elevation_loss_m`, `net_grade_percent`. Como o arquivo SRTM armazena altitudes inteiras de 16 bits em metros (`227 m` a `325 m`, `22.941` pontos no cache local), o passo espacial mínimo de `50 m` elimina o ruído altimétrico de GPS, enquanto o *deadband* vertical de `0,5 m` atua como salvaguarda preventiva para eventuais leituras fracionárias ou interpoladas.
  3. *Clima & Conforto Térmico (2):* `aux_hvac_energy_kwh`, `mean_oat_c`.
  4. *Eletroquímica da Bateria (1):* `initial_soc_percent`.
* **Dois Alvos Preditos Independentemente (nas `482` viagens válidas):**
  * **Alvo 1 — Energia Líquida (`energy_net_kwh`, em kWh):** `0,0034` a `3,7278 kWh` (média `0,7485 kWh`).
  * **Alvo 2 — Queda do Estado de Carga (`delta_soc_percent`, em %):** `0,0000` a `22,8000 %` (média `4,2930 %`).
* **Métricas Avaliadas:** `R²`, `MAE`, `RMSE`, `MAPE` (filtrado para `Elíq ≥ 0,05 kWh` ou `ΔSOC ≥ 0,5%`), `MAE/100 km`, diagnóstico de resíduos (`média`, `desvio-padrão`, `mediana`, `P5–P95`), contagem de **previsões negativas** (violação de limite físico) e interpretabilidade via **SHAP** (`LinearExplainer`, `TreeExplainer` e `KernelExplainer`).

---

## 2. Modelos Implementados e Evolução Experimental

### 2.1 Piso de Sanidade (`DummyRegressor`) e Baseline Principal (`Regressão Linear Múltipla`)
* **`DummyRegressor(strategy="mean")`:** prevê sempre a média do treino (`R² = −0,0049`, `MAE = 0,4851 kWh` na Energia; `R² = −0,0339`, `MAE = 3,1158 %` no `ΔSOC`). Serve apenas como referência mínima de sanidade.
* **Regressão Linear Múltipla (`LinearRegression()` do `scikit-learn`):** adotada como **baseline científico principal**, cumprindo na Fase 2 o papel de linha de base equivalente ao `K-Means` na Fase 1.
  * Ajusta a equação fechada $y = \beta_0 + \sum_{j=1}^{12}\beta_j x_j$ sem hiperparâmetros nem aleatoriedade.
  * Em `src/xgboost_consumption_model.py`, foi integrada ao mesmo `5-Fold Stratified CV` das `385` viagens de treino, com extração de coeficientes físicos ($\beta$), `SHAP LinearExplainer` (usando todas as `385` amostras de fundo) e 3 gráficos diagnósticos individuais (`36_`, `37_`, `38_`).
  * Na Energia Líquida, apresentou intercepto de `−0,540241 kWh` e coeficientes fisicamente coerentes (ex.: `+0,074363 kWh/km` em `distance_km`, `+0,006520 kWh/m` em `elevation_gain_m`, `−0,004370 kWh/m` em `elevation_loss_m` e `+1,267352 × aux_hvac_energy_kwh`). No `ΔSOC`, apresentou intercepto de `−0,455829 %`, `+8,244671 %/kWh` em `aux_hvac_energy_kwh`, `+0,302513 %/km` em `distance_km` e `−0,036280 %/°C` em `mean_oat_c`.

### 2.2 `XGBoost Regressor`
* Implementado em `src/xgboost_consumption_model.py` com parâmetros conservadores voltados à prevenção de *overfitting* em `385` viagens: `n_estimators=120`, `learning_rate=0.04`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.85`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=42`.
* Explicabilidade exata via `shap.TreeExplainer` e gráficos `25_`, `26_`, `27_`.

### 2.3 `MLP Regressor` (Desenvolvimento Estruturado em 3 Etapas)
Implementado em `src/mlp_consumption_model.py` usando `Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(...))])` (garantindo padronização Z-score ajustada estritamente nos dados de treino de cada *fold*, sem *data leakage*) e `solver="lbfgs"` (otimizador quase-Newton mais estável e rápido que `adam` para bases tabulares pequenas abaixo de 1.000 amostras).

#### Etapa 1 — Diagnóstico Base (`(16,)`, `relu`, `lbfgs`, `alpha = 0,1`, `max_iter = 2000`)
Executada na base saneada (`385` viagens de treino):
* **Energia Líquida:** convergiu em `892` iterações (`0` avisos na CV e no treino completo), `Treino R² = 0,9856`, `CV R² = 0,9445 ± 0,0172` (`gap = +0,0411`, `1` previsão negativa *out-of-fold*: `−0,0694 kWh`).
* **`ΔSOC`:** apresentou ***overfitting* severo e falha de convergência** — `Treino R² = 0,9784`, mas `CV R² = 0,6740 ± 0,0998` (`gap = +0,3044`), `3` avisos `ConvergenceWarning` na CV + `1` no treino completo (atingindo o limite de `2.000` iterações) e **`24` previsões negativas *out-of-fold*** (até `−2,4618 %`).
* **Causa matemática identificada:** na implementação do `scikit-learn` (`MLPRegressor._backprop`), a penalidade L2 somada à perda quadrática é $\frac{\alpha}{2n}\|W\|_2^2$. Como o `ΔSOC` (média `4,29%`, até `22,80%`) tem escala ~`5,7×` maior que a Energia (`0,75 kWh`, até `3,73 kWh`), o erro quadrático é ~`33×` maior; logo, `alpha = 0,1` segurou a Energia mas foi fraco demais para regularizar o `ΔSOC`.

#### Etapas 2 e 2b — Seleção de Arquitetura, Regularização L2 (`alpha`) e Validação do Pico na CV 5-Fold
* Na busca controlada de `36` combinações por alvo (`hidden_layer_sizes ∈ {(8,), (16,), (16, 8)}`, `activation ∈ {relu, tanh}`, `alpha ∈ {0,1; 0,5; 1,0; 5,0; 10,0; 20,0}`, `max_iter = 5000`), seguida do refinamento comparativo entre manter `y` em escala original (**Opção 1**) vs. padronizar `y` via `TransformedTargetRegressor` (**Opção 2**), adotou-se a **Opção 1 (sem padronizar `y`)** com a arquitetura enxuta em funil `(16, 8)` (`353` parâmetros: `12 → 16 → 8 → 1`) para ambos os alvos:
  * **Energia Líquida:** `(16, 8)` + `relu` + `alpha = 1,0` → `CV R² = 0,9625 ± 0,0204` (`301` iterações no treino completo, `0` avisos).
  * **`ΔSOC`:** `(16, 8)` + `tanh` + `alpha = 20,0` → `CV R² = 0,8706 ± 0,0182` (`1.063` iterações no treino completo, `0` avisos).
* **Verificação da curva de regularização após o filtro OBD (`482` viagens):** para verificar se a remoção das `4` viagens com lacuna OBD deslocaria o ponto ótimo de `alpha` no `ΔSOC`, a curva de CV 5-Fold para `(16, 8)` + `tanh` foi reavaliada na base saneada:
  * `alpha = 1,0` (`0,7059`) → `5,0` (`0,7907`) → `10,0` (`0,8492`) → `15,0` (`0,8658`) → **`20,0` (`0,8706`, pico exato confirmado)** → `30,0` (`0,8665`).
  * Esse comportamento confirma que o `alpha = 20,0` não era artefato das `4` viagens corrompidas, mas sim consequência direta da escala quadrática do `ΔSOC` (`~33×` maior que a da energia em `kWh`) combinada ao ruído intrínseco de quantização do BMS.

#### Etapa 3 — Avaliação Final no *Holdout* (`97` viagens), `SHAP KernelExplainer` e Gráficos
Treino final das duas configurações selecionadas nas `385` viagens, avaliação única nas `97` viagens inéditas, explicabilidade com `shap.KernelExplainer` (`50` centróides *k-means* de fundo, `nsamples=200`, `seed=42`) e geração das figuras `39_`, `40_`, `41_`, além dos painéis comparativos dos três modelos (`42_` e `43_`).

---

## 3. Resultados Consolidados (`482` Viagens): Regressão Linear × XGBoost × MLP

### 3.1 Alvo 1 — Energia Líquida da Bateria (`energy_net_kwh`)

| Modelo | CV 5-Fold R² (385) | CV MAE | CV RMSE | **Teste R² (97)** | **Teste MAE** | **Teste RMSE** | **Teste MAPE** | **MAE / 100 km** | Negativos no Teste |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy (Média — sanidade) | — | — | — | `−0,0049` | `0,4851 kWh` | `0,6480 kWh` | `108,17%` | `16,75 kWh` | `0` |
| **Regressão Linear (Baseline)** | `0,9526 ± 0,0193` | `0,0914 kWh` | `0,1310 kWh` | `0,9615` | `0,0932 kWh` | `0,1268 kWh` | `18,83%` | `3,01 kWh` | `1` (`−0,0471`) |
| **XGBoost** | `0,9213 ± 0,0340` | `0,1077 kWh` | `0,1669 kWh` | `0,9228` | `0,1131 kWh` | `0,1796 kWh` | `17,78%` | `2,85 kWh` | `0` (`mín. +0,1176`) |
| **MLP (`(16, 8)`, `relu`, `α=1,0`)** | **`0,9625 ± 0,0204`** | **`0,0763 kWh`** | **`0,1155 kWh`** | **`0,9757`** | **`0,0790 kWh`** | **`0,1008 kWh`** | **`16,75%`** | **`2,72 kWh`** | **`0`** (`mín. +0,0559`) |

* **Resíduos no Teste (`97` viagens):**
  * Regressão Linear: média `−0,0205 kWh`, desvio `0,1251 kWh`, mediana `−0,0235 kWh`, `P5–P95 = [−0,2042, +0,1839] kWh`.
  * XGBoost: média `−0,0192 kWh`, desvio `0,1785 kWh`, mediana `−0,0236 kWh`, `P5–P95 = [−0,2759, +0,2283] kWh`.
  * **MLP:** média **`−0,0175 kWh`**, desvio **`0,0993 kWh`**, mediana **`−0,0228 kWh`**, `P5–P95 = [−0,1576, +0,1627] kWh`.

### 3.2 Alvo 2 — Queda do Estado de Carga (`delta_soc_percent`)

| Modelo | CV 5-Fold R² (385) | CV MAE | CV RMSE | **Teste R² (97)** | **Teste MAE** | **Teste RMSE** | **Teste MAPE** | **MAE / 100 km** | Negativos no Teste |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy (Média — sanidade) | — | — | — | `−0,0339` | `3,1158 %` | `4,3636 %` | `89,36%` | `95,20 %` | `0` |
| **Regressão Linear (Baseline)** | `0,8016 ± 0,0505` | `1,2746 %` | `1,6687 %` | `0,8727` | `1,1708 %` | `1,5314 %` | `35,02%` | `32,59 %` | `1` (`−0,5533`) |
| **XGBoost** | `0,8460 ± 0,0268` | `1,0505 %` | `1,4780 %` | `0,8479` | `1,1819 %` | `1,6734 %` | `30,72%` | `31,17 %` | `1` (`−0,3175`) |
| **MLP (`(16, 8)`, `tanh`, `α=20,0`)** | **`0,8706 ± 0,0182`** | **`1,0318 %`** | **`1,3608 %`** | **`0,8897`** | **`1,0642 %`** | **`1,4252 %`** | **`27,80%`** | **`30,00 %`** | **`0`** (`mín. +0,2633`) |

* **Resíduos no Teste (`97` viagens):**
  * Regressão Linear: média `+0,3018 %`, desvio `1,5014 %`, mediana `+0,2350 %`, `P5–P95 = [−1,9804, +2,7760] %`.
  * XGBoost: média `+0,2800 %`, desvio `1,6498 %`, mediana `+0,1193 %`, `P5–P95 = [−1,8628, +2,9942] %`.
  * **MLP:** média **`+0,2364 %`**, desvio **`1,4055 %`**, mediana **`+0,0185 %`**, `P5–P95 = [−1,8231, +2,3900] %`.

### 3.3 Consenso Físico de Importância das Variáveis (Top 5 SHAP no Teste)

| Pos. | Regressão Linear (`LinearExplainer`) | XGBoost (`TreeExplainer`) | MLP (`KernelExplainer`) |
| :---: | :--- | :--- | :--- |
| **Energia 1º** | `elevation_gain_m` (`0,2924 kWh`, `+`) | `elevation_gain_m` (`0,2444 kWh`) | `distance_km` (`0,2129 kWh`) |
| **Energia 2º** | `distance_km` (`0,2084 kWh`, `+`) | `distance_km` (`0,1383 kWh`) | `aux_hvac_energy_kwh` (`0,1413 kWh`) |
| **Energia 3º** | `elevation_loss_m` (`0,2001 kWh`, `−`) | `aux_hvac_energy_kwh` (`0,1151 kWh`) | `elevation_gain_m` (`0,1132 kWh`) |
| **Energia 4º** | `aux_hvac_energy_kwh` (`0,1497 kWh`, `+`) | `duration_s` (`0,0385 kWh`) | `duration_s` (`0,0671 kWh`) |
| **Energia 5º** | `duration_s` (`0,0555 kWh`, `+`) | `mean_oat_c` (`0,0309 kWh`) | `mean_oat_c` (`0,0574 kWh`) |
| **`ΔSOC` 1º** | `aux_hvac_energy_kwh` (`0,9736 %`, `+`) | `distance_km` (`0,8963 %`) | `duration_s` (`1,0750 %`) |
| **`ΔSOC` 2º** | `distance_km` (`0,8478 %`, `+`) | `aux_hvac_energy_kwh` (`0,8896 %`) | **`initial_soc_percent` (`0,9141 %`)** |
| **`ΔSOC` 3º** | `elevation_gain_m` (`0,7279 %`, `+`) | `elevation_gain_m` (`0,7197 %`) | `aux_hvac_energy_kwh` (`0,9113 %`) |
| **`ΔSOC` 4º** | `duration_s` (`0,6735 %`, `+`) | `duration_s` (`0,5666 %`) | `elevation_gain_m` (`0,6939 %`) |
| **`ΔSOC` 5º** | `mean_oat_c` (`0,3681 %`, `−`) | `initial_soc_percent` (`0,4803 %`) | `distance_km` (`0,5412 %`) |

---

## 4. Conclusões Principais: O que os Resultados Respondem

### 4.1 É possível fazer a predição do consumo energético?
**Sim — com altíssima precisão para energia líquida e excelente precisão operacional para queda percentual de bateria, utilizando apenas 12 atributos físicos sobre a base saneada.**
* Na **Energia Líquida**, o `MLP` alcançou **`R² = 0,9757`**, **`MAE = 0,0790 kWh`** (`79,0 Wh` por viagem; `2,72 kWh/100 km`) e `RMSE = 0,1008 kWh`. Para a bateria nominal de `24 kWh` do Nissan Leaf, esse erro médio equivale a cerca de **0,33% da capacidade total da bateria**.
* Na **queda de carga (`ΔSOC`)**, após a remoção das `4` viagens com apagão de gravação OBD, o `MLP` atingiu **`R² = 0,8897`**, **`MAE = 1,0642 p.p.`** e `RMSE = 1,4252 p.p.` (com mediana de resíduo quase nula: `+0,0185 p.p.` e **zero previsões negativas**), fornecendo estimativa muito confiável mesmo sob os arredondamentos do BMS e amplitudes térmicas severas (`−14,45 °C` a `35,51 °C`).
* **Ressalva fundamental de aplicabilidade:** os experimentos atuais utilizam a energia auxiliar **medida** de climatização (`aux_hvac_energy_kwh`) e trajetos urbanos/periurbanos de até `21,5 km` de um único modelo de veículo elétrico. Para uso estritamente **pré-partida**, é necessária uma etapa prévia de estimativa térmica do HVAC.

### 4.2 O que a comparação entre os três algoritmos revela?
1. **Qualidade de dados de telemetria impacta mais o modelo do que qualquer ajuste fino isolado:**
   * A identificação e remoção das apenas `4` viagens (`0,8%` da base) com lacunas de gravação OBD acima de 10 minutos elevou o `R²` de teste do `ΔSOC` em `+8,97 p.p.` no MLP, `+11,43 p.p.` no XGBoost e `+18,21 p.p.` na Regressão Linear, além de reduzir em mais de `3×` o desvio-padrão entre *folds* na CV do MLP (`±0,0580 → ±0,0182`).
2. **A física do consumo em `kWh` tem forte componente aditivo/contínuo, mas interações não lineares ainda agregam ganho claro:**
   * A **Regressão Linear** já explica `96,15%` da Energia Líquida (`CV = 0,9526`) e `87,27%` do `ΔSOC` no teste (`CV = 0,8016`), confirmando o peso somativo de distância, relevo (subidas menos recuperação regenerativa nas descidas) e HVAC. Contudo, ainda perde consistência na validação cruzada do `ΔSOC` (`0,8016 ± 0,0505`) e gera previsões fisicamente inválidas em viagens curtas (`1` valor negativo em cada alvo).
3. **Árvores (`XGBoost`) capturam bem a não linearidade do `ΔSOC` na CV, mas sofrem com partições em degrau em bases pequenas:**
   * Na CV 5-Fold do `ΔSOC`, o **XGBoost** superou com folga a Regressão Linear (`0,8460 ± 0,0268` vs. `0,8016 ± 0,0505`, `MAE 1,0505 %` vs. `1,2746 %`). Entretanto, em `385` viagens com relações suaves e contínuas, árvores de decisão produzem patamares discretos e não extrapolam além do teto das folhas vistas no treino — razão pela qual ficou atrás da Linear e do MLP na Energia Líquida (`CV R² = 0,9213`, `Teste R² = 0,9228`) e apresentou maior dispersão nos extremos do `ΔSOC` no teste (`RMSE = 1,6734 %`, `1` previsão negativa de `−0,3175 %`).
4. **O `MLP` enxuto e regularizado venceu em todas as métricas principais nos dois alvos:**
   * Uniu a **continuidade suave** (superando o teto de degraus das árvores na Energia Líquida: `MAE` `30,1%` menor que o XGBoost, `0,0790` vs. `0,1131 kWh`, e `15,2%` menor que a Linear) com a **modelagem não linear** das duas camadas ocultas `(16, 8)` (liderando o `ΔSOC` tanto na CV `0,8706 ± 0,0182` quanto no teste `R² = 0,8897`, `MAE = 1,0642 %`, `RMSE = 1,4252 %`, com **zero previsões negativas** em ambos os alvos).
5. **Lição metodológica sobre redes neurais em dados tabulares pequenos:**
   * O contraste entre a Etapa 1 (`ΔSOC CV R² = 0,6740`, `4` avisos de convergência, `24` negativos) e o modelo final regularizado (`CV R² = 0,8706`, `Teste R² = 0,8897`, `0` avisos, `0` negativos) comprova que redes neurais em ~400 amostras **exigem** padronização em `Pipeline`, arquitetura compacta, otimizador adequado (`lbfgs`) e regularização L2 calibrada à escala do alvo na CV.
6. **Consenso físico independente do algoritmo:**
   * Os três modelos concordam que **distância, duração, ganho acumulado em aclives (SRTM) e climatização (HVAC) / temperatura externa** governam o consumo líquido, enquanto o **estado de carga inicial (`initial_soc_percent`)** assume papel central na explicação do `ΔSOC` (2º lugar no SHAP do MLP com `0,9141 %` e 5º no XGBoost com `0,4803 %`), refletindo a eletroquímica da bateria e a limitação da frenagem regenerativa próxima de 100% de carga.

---

## 5. Qual Algoritmo Escolher para um Sistema Futuro em Tempo Real?

### 5.1 Recomendação Principal: `MLP Regressor` com Trava Física `max(0, y_pred)`
Entre os modelos testados, o **`MLP`** é o melhor candidato para embarcar em uma aplicação futura de estimativa de consumo e autonomia:
1. **Maior acurácia e consistência nos dois alvos** (`79,0 Wh` de erro médio na energia; `1,06 p.p.` no `ΔSOC`; menor dispersão entre *folds* na CV; zero previsões negativas no teste).
2. **Custo computacional mínimo na inferência em tempo real:** depois de treinado, o modelo `(16, 8)` resume-se a apenas **353 pesos** (três matrizes pequenas: `12×16`, `16×8`, `8×1`) somados a `24` constantes do `StandardScaler` (`12` médias e `12` desvios). A inferência ocorre em **microssegundos** em CPU comum, smartphone ou módulo embarcado, sem necessidade de GPU.
3. **Respostas suaves no painel:** por gerar uma função contínua, pequenas alterações na rota ou na temperatura atualizam a autonomia estimada sem "saltos" abruptos entre faixas de árvores.
4. **Salvaguarda simples de produção:** mesmo já apresentando `0` previsões negativas no conjunto de teste saneado, aplicar `max(0.0, y_pred)` na camada de saída garante conformidade física em qualquer cenário extremo de produção.

### 5.2 Papéis Complementares dos Outros Dois Algoritmos
* **`XGBoost`:** continua sendo uma excelente alternativa caso o sistema futuro precise lidar diretamente com **sensores falhando (`NaN`)**, incorporar **muitas variáveis categóricas heterogêneas** (múltiplos modelos/marcas de veículos, tipos de via/pneu) ou priorizar pipelines menos sensíveis à calibração de escala e regularização.
* **`Regressão Linear Múltipla`:** útil como **modelo de contingência ultraleve (*fallback*)** (apenas 13 coeficientes) e como referência linear de auditoria em rotas muito longas.

### 5.3 Arquitetura Sugerida para Operação em Tempo Real
Mesmo treinado em nível de viagem completa, o modelo pode operar em tempo real em **dois modos integrados**:
1. **Modo Pré-Partida (Planejamento de Rota):** recebe distância, duração prevista e perfil altimétrico (SRTM/API de relevo) da rota traçada no GPS, temperatura externa prevista, SOC atual da bateria e uma estimativa prévia de demanda do HVAC (calculada a partir da temperatura externa e da duração da rota).
2. **Modo Recálculo Dinâmico em Percurso (ex.: a cada 1 minuto ou 1 km):** alimenta o modelo com o **trecho restante** da rota, atualizando em tempo real o SOC atual e a potência real de HVAC já acionada pelo condutor, recalculando continuamente a autonomia restante até o destino.
3. **Tratamento de Rotas Longas (> 21,5 km):** enquanto o modelo não for retreinado com dados rodoviários extensos, rotas longas podem ser divididas em segmentos de `5` a `15 km` (dentro do domínio de treino), somando-se o consumo previsto de cada segmento.

---

## 6. Próximos Passos e Evoluções Futuras do Projeto

1. **Estimador Térmico Pré-Partida (HVAC):** desenvolver um submódulo auxiliar para estimar `aux_hvac_energy_kwh` usando apenas temperatura externa, horário/estação e duração estimada da rota, viabilizando previsão 100% *antes da partida*.
2. **Integração do Fator Humano (Fase 1 + Fase 2):** incorporar o perfil de condução histórico do motorista (econômico, moderado ou agressivo, classificado na Fase 1 via DTW/Louvain/K-DBA/K-Means) como variável explicativa adicional no modelo preditivo da Fase 2, quantificando o ganho na personalização da autonomia.
3. **Ampliação do Domínio de Dados:** validar e retreinar os modelos em frotas heterogêneas (diferentes veículos elétricos, capacidades de bateria e climas) e em trajetos rodoviários de longa distância.
4. **Expansão Opcional de Algoritmos e Abordagem Microscópica:**
   * No nível de viagem (*trip-level*), testar algoritmos adicionais como **Random Forest** (*bagging* de árvores) e **SVR com kernel RBF** (máquina de kernel) para ampliar ainda mais o benchmark comparativo.
   * No nível instantâneo (`1 Hz`), investigar redes sequenciais (**LSTM / GRU / CNN-1D**) voltadas especificamente à predição de potência instantânea $P(t)$ segundo a segundo.

---

## 7. Rastreabilidade de Código, Resultados e Figuras

* **Scripts principais:**
  * `src/trip_feature_extraction.py` (extração de viagens a partir da base `1 Hz`, integração energética com frenagem regenerativa, altimetria NASA SRTM 30 m e filtro de integridade `dur >= 60 s`, `dist >= 0,5 km`, `max_obd_gap_s <= 60 s`).
  * `src/xgboost_consumption_model.py` (Dummy, Regressão Linear Baseline + CV + SHAP, XGBoost + CV + SHAP e figuras `25_`–`27_`, `35_`–`38_`).
  * `src/mlp_consumption_model.py` (MLP em 3 etapas + refinamento `2b` via `--stage 1 | 2 | 2b | 3`, figuras `39_`–`41_` e painéis comparativos `42_`–`43_`).
* **Relatórios numéricos (`outputs/results/trips/`):**
  * `ev_trips_summary_statistics.json` (estatísticas da base consolidada e contagem explícita das `18` viagens curtas e `4` viagens com lacuna OBD `> 60 s`).
  * `linear_regression_baseline_evaluation.json`
  * `xgboost_dual_model_evaluation.json`, `xgboost_dual_model_evaluation.csv`, `xgboost_shap_rankings.csv`
  * `mlp_etapa1_diagnostic.json`, `mlp_etapa2_cv_tuning.json`, `mlp_etapa2_cv_tuning.csv`, `mlp_etapa2b_refinement.json`, `mlp_etapa2b_refinement.csv`
  * `mlp_dual_model_evaluation.json`, `mlp_shap_rankings.csv`
  * `phase2_three_models_comparison.csv`
* **Modelos serializados (`outputs/models/`):**
  * `xgboost_energy_net_model.joblib`, `xgboost_delta_soc_model.joblib`
  * `mlp_energy_net_model.joblib`, `mlp_delta_soc_model.joblib`
* **Figuras geradas (`outputs/figures/trips/`):**
  * Dicionário de Dados (`48` variáveis, `482` viagens válidas): `28_dicionario_variaveis_ev_trips.png`
  * XGBoost: `25_xgboost_real_vs_pred_energy.png`, `26_xgboost_residuals_distribution.png`, `27_xgboost_feature_importance_shap.png`
  * Regressão Linear: `36_linear_regression_real_vs_pred_energy.png`, `37_linear_regression_residuals_distribution.png`, `38_linear_regression_feature_importance_shap.png`
  * MLP: `39_mlp_real_vs_pred_energy.png`, `40_mlp_residuals_distribution.png`, `41_mlp_feature_importance_shap.png`
  * Comparativos: `35_comparacao_modelos_baselines_xgboost.png`, `42_comparacao_3_modelos_barras_cv_teste.png`, `43_comparacao_3_modelos_dispersao_barras.png`
