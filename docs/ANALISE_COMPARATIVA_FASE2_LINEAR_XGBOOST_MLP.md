# Análise Comparativa da Fase 2: Regressão Linear (Baseline), XGBoost e MLP na Predição de Consumo Energético de Veículos Elétricos

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** IFSULDEMINAS – Campus Poços de Caldas  
**Base de Dados:** Vehicle Energy Dataset (VED) – 3 unidades do Nissan Leaf 2013 (`VehId 10, 455, 541`, bateria nominal de `24 kWh`)  
**Objetivo do Documento:** Registrar de forma completa a implementação do baseline e do modelo MLP na Fase 2, consolidar os resultados obtidos frente ao XGBoost, sintetizar as conclusões científicas e técnicas e estruturar recomendações para uma eventual implementação futura em tempo real.

---

## 1. Contexto e Protocolo Experimental Unificado

Após a revisão da versão preliminar do TCC pelo orientador, surgiu a recomendação de incluir um **baseline claro** e **outros algoritmos preditivos** na Fase 2 para avaliar de forma comparativa o ganho real de cada abordagem frente ao `XGBoost`.

Para garantir comparação estritamente justa (zero viés de divisão e zero *data leakage*), todos os modelos foram avaliados sob o **mesmo protocolo experimental**:

* **Base consolidada:** `data/processed/trips/ev_trips_complete_dataset.parquet` (`504` viagens registradas; **`486` viagens válidas** após o filtro físico `duração ≥ 60 s` e `distância ≥ 0,5 km`, cobrindo trajetos de `0,51 km` a `21,50 km`, média de `4,82 km` e `9,32 min`).
* **Por que modelagem em nível de viagem (*trip-level*) e não segundo a segundo (`1 Hz`)?**
  * Todos os `476.308` registros de `1 Hz` foram usados na etapa de extração (`src/trip_feature_extraction.py`) para integrar distância, acelerações, variabilidade de velocidade, relevo SRTM, energia de HVAC e balanço energético com frenagem regenerativa.
  * Consolidar em 1 linha por viagem completa é o padrão adequado para modelos tabulares (`Regressão Linear`, `XGBoost` e `MLP`), pois: (i) viagens têm durações muito diferentes e esses algoritmos exigem entrada de dimensão fixa (`12` atributos); (ii) o alvo `ΔSOC` varia em degraus discretos no painel (sendo `0%` na maioria dos segundos individuais); (iii) impede vazamento de segundos vizinhos entre treino e teste; e (iv) reflete o uso prático em estimativa de autonomia de rota.
* **Divisão Treino / Teste (*Holdout*) e Embaralhamento:** `train_test_split(..., test_size=0.20, random_state=42, stratify=pd.qcut(distance_km, q=5))` (`shuffle=True` padrão), separando deterministicamente em memória **`388` viagens de treino (80%)** e **`98` viagens inéditas de teste (20%)** com mesma proporção de trajetos curtos e longos.
* **Validação Cruzada Interna:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` estratificada em 5 quantis de distância dentro das `388` viagens de treino. No desenvolvimento do MLP, o conjunto de teste (*holdout*, `98` viagens) permaneceu **completamente isolado** durante todo o diagnóstico (Etapa 1) e ajuste de hiperparâmetros (Etapas 2 e 2b), sendo aberto apenas na avaliação final (Etapa 3).
* **12 Variáveis de Entrada (`FEATURE_COLS`):**
  1. *Cinemática & Operação (6):* `distance_km`, `duration_s`, `mean_speed_kmh`, `std_speed_kmh`, `mean_pos_accel_ms2`, `mean_neg_accel_ms2`.
  2. *Topografia — NASA SRTM 30 m com passo mínimo de 50 m e deadband de 0,5 m (3):* `elevation_gain_m`, `elevation_loss_m`, `net_grade_percent`.
  3. *Clima & Conforto Térmico (2):* `aux_hvac_energy_kwh`, `mean_oat_c`.
  4. *Eletroquímica da Bateria (1):* `initial_soc_percent`.
* **Dois Alvos Preditos Independentemente:**
  * **Alvo 1 — Energia Líquida (`energy_net_kwh`, em kWh):** varia de `0,0034` a `3,7278 kWh` (média `0,7548 kWh`).
  * **Alvo 2 — Queda do Estado de Carga (`delta_soc_percent`, em %):** varia de `0,0` a `22,93 %` (média `4,4183 %`).
* **Métricas Avaliadas:** `R²`, `MAE`, `RMSE`, `MAPE` (filtrado para `Elíq ≥ 0,05 kWh` ou `ΔSOC ≥ 0,5%`), `MAE/100 km`, diagnóstico de resíduos (`média`, `desvio-padrão`, `mediana`, `P5–P95`), contagem de **previsões negativas** (violação de limite físico) e interpretabilidade via **SHAP** (`LinearExplainer`, `TreeExplainer` e `KernelExplainer`).

---

## 2. Modelos Implementados e Evolução Experimental

### 2.1 Piso de Sanidade (`DummyRegressor`) e Baseline Principal (`Regressão Linear Múltipla`)
* **`DummyRegressor(strategy="mean")`:** prevê sempre a média do treino (`R² = −0,0145`, `MAE = 0,4036 kWh` na Energia; `R² = −0,0020`, `MAE = 2,6866 %` no `ΔSOC`). Serve apenas como referência mínima de sanidade.
* **Regressão Linear Múltipla (`LinearRegression()` do `scikit-learn`):** adotada como **baseline científico principal**, cumprindo na Fase 2 o papel de linha de base equivalente ao `K-Means` na Fase 1.
  * Ajusta a equação fechada $y = \beta_0 + \sum_{j=1}^{12}\beta_j x_j$ sem hiperparâmetros ou aleatoriedade.
  * Em `src/xgboost_consumption_model.py`, foi integrada ao mesmo `5-Fold Stratified CV` do treino, com extração de coeficientes físicos ($\beta$), `SHAP LinearExplainer` (usando todas as `388` amostras de fundo) e 3 gráficos diagnósticos individuais (`36_`, `37_`, `38_`).
  * Na Energia Líquida, apresentou intercepto de `−0,548405 kWh` e coeficientes fisicamente coerentes (ex.: `+0,074215 kWh/km` em `distance_km`, `+0,00626 kWh/m` em `elevation_gain_m`, `−0,003674 kWh/m` em `elevation_loss_m` e `+1,354399 × aux_hvac_energy_kwh`).

### 2.2 `XGBoost Regressor`
* Implementado em `src/xgboost_consumption_model.py` com parâmetros conservadores voltados à prevenção de *overfitting* em `388` viagens: `n_estimators=120`, `learning_rate=0.04`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.85`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=42`.
* Explicabilidade exata via `shap.TreeExplainer` e gráficos `25_`, `26_`, `27_`.

### 2.3 `MLP Regressor` (Desenvolvimento Estruturado em 3 Etapas)
Implementado em `src/mlp_consumption_model.py` usando `Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(...))])` (garantindo padronização Z-score ajustada estritamente nos dados de treino de cada *fold*, sem *data leakage*) e `solver="lbfgs"` (otimizador quase-Newton mais estável e rápido que `adam` para bases tabulares pequenas abaixo de 1.000 amostras).

#### Etapa 1 — Diagnóstico Base (`(16,)`, `relu`, `lbfgs`, `alpha = 0,1`, `max_iter = 2000`)
* **Energia Líquida:** convergiu em `691` iterações (`0` avisos), `Treino R² = 0,9868`, `CV R² = 0,9538 ± 0,0142` (`MAE = 0,0924 kWh`, `gap = +0,0330`, `1` previsão negativa *out-of-fold*: `−0,0369 kWh`).
* **`ΔSOC`:** apresentou ***overfitting* severo e falha de convergência** — `Treino R² = 0,9770`, mas `CV R² = 0,6451 ± 0,1022` (`MAE = 1,7295 %`, `gap = +0,3319`), **`5` avisos `ConvergenceWarning`** (todos os 5 *folds* atingiram `2.000` iterações) e **`25` previsões negativas *out-of-fold*** (até `−2,0621 %`).
* **Causa matemática identificada:** na implementação do `scikit-learn` (`MLPRegressor._backprop`), a penalidade L2 somada à perda quadrática é $\frac{\alpha}{2n}\|W\|_2^2$. Como o `ΔSOC` (média `4,42%`, até `22,93%`) tem escala ~`5,9×` maior que a Energia (`0,75 kWh`, até `3,73 kWh`), o erro quadrático é ~`34×` maior; logo, `alpha = 0,1` segurou a Energia mas foi fraco demais para regularizar o `ΔSOC`.

#### Etapa 2 — Ajuste Controlado na CV 5-Fold (`36` combinações por alvo, `max_iter = 5000`)
Grade avaliada exclusivamente nas `388` viagens de treino: `hidden_layer_sizes ∈ {(8,), (16,), (16, 8)}`, `activation ∈ {relu, tanh}`, `alpha ∈ {0,1; 0,5; 1,0; 5,0; 10,0; 20,0}`.
* **Energia Líquida:** atingiu pico interior claro em **`(16, 8)` + `relu` + `alpha = 1,0`** (`CV R² = 0,9681 ± 0,0175`, `MAE = 0,0723 kWh`, `RMSE = 0,1111 kWh`, `gap = +0,0079`, `214` iterações, `0` avisos, `0` negativos OOF).
* **`ΔSOC`:** saltou para **`CV R² = 0,8523 ± 0,0580`** (`MAE = 1,1158 %`, `RMSE = 1,5818 %`, `gap = +0,0902`, `567` iterações, `0` avisos, `0` negativos OOF) em **`(16, 8)` + `tanh` + `alpha = 20,0`**.

#### Etapa 2b — Refinamento e Validação do Pico (`Opção 1` vs. `Opção 2`, na CV 5-Fold)
Para verificar se `alpha = 20,0` no `ΔSOC` era o pico real (ou se continuaria subindo acima da borda da grade) e testar o efeito de padronizar também o alvo `y`:
* **Opção 1 (estender `alpha` até `100,0` no `ΔSOC` sem padronizar `y`):** confirmou **`alpha = 20,0` como o pico interior exato** para `(16, 8)` + `tanh`:
  * `alpha = 10,0` (`0,8231`) → **`20,0` (`0,8523`)** → `30,0` (`0,8473`) → `40,0` (`0,8338`) → `50,0` (`0,8030`) → `75,0` (`0,7498`) → `100,0` (`0,7408`).
* **Opção 2 (`TransformedTargetRegressor` padronizando `X` e `y`):** levou o `alpha` ótimo para `2,0` nos dois alvos com `0` avisos em todas as `72` combinações, porém obteve `CV R²` e `RMSE` ligeiramente inferiores e gerou `8` previsões negativas no `ΔSOC`:
  * Energia Líquida (`(16,)`, `relu`, `α = 2,0`): `CV R² = 0,9679`, `MAE = 0,0724 kWh`, `0` negativos.
  * `ΔSOC` (`(8,)`, `tanh`, `α = 2,0`): `CV R² = 0,8475`, `MAE = 1,1074 %`, `8` negativos OOF (`mín. −0,8295 %`).
* **Decisão metodológica:** adotou-se a **Opção 1 (sem padronizar `y`)**, mantendo a mesma arquitetura enxuta em funil `(16, 8)` (`353` parâmetros: `12 → 16 → 8 → 1`) para ambos os alvos, com melhor `CV R²`, melhor `CV RMSE`, `0` avisos de convergência e `0` previsões negativas OOF.

#### Etapa 3 — Avaliação Final no *Holdout* (`98` viagens), `SHAP KernelExplainer` e Gráficos
Treino final das duas configurações selecionadas nas `388` viagens, teste único nas `98` viagens inéditas, explicabilidade com `shap.KernelExplainer` (`50` centróides *k-means* de fundo, `nsamples=200`, `seed=42`) e geração das figuras `39_`, `40_` e `41_`.

---

## 3. Resultados Consolidados: Regressão Linear × XGBoost × MLP

### 3.1 Alvo 1 — Energia Líquida da Bateria (`energy_net_kwh`)

| Modelo | CV 5-Fold R² (388) | CV MAE | CV RMSE | **Teste R² (98)** | **Teste MAE** | **Teste RMSE** | **Teste MAPE** | **MAE / 100 km** | Negativos no Teste |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy (Média — sanidade) | — | — | — | `−0,0145` | `0,4036 kWh` | `0,5095 kWh` | `106,16%` | `17,49 kWh` | `0` |
| **Regressão Linear (Baseline)** | `0,9566 ± 0,0173` | `0,0894 kWh` | `0,1302 kWh` | `0,9295` | `0,1018 kWh` | `0,1343 kWh` | `19,00%` | `3,22 kWh` | `1` (`−0,0228`) |
| **XGBoost** | `0,9192 ± 0,0146` | `0,1115 kWh` | `0,1816 kWh` | `0,9300` | `0,0985 kWh` | `0,1339 kWh` | `18,65%` | `3,26 kWh` | `0` |
| **MLP (`(16, 8)`, `relu`, `α=1,0`)** | **`0,9681 ± 0,0175`** | **`0,0723 kWh`** | **`0,1111 kWh`** | **`0,9616`** | **`0,0738 kWh`** | **`0,0992 kWh`** | **`12,64%`** | **`2,18 kWh`** | **`0`** (`mín. +0,1046`) |

* **Resíduos no Teste (`98` viagens):**
  * Regressão Linear: média `−0,0077 kWh`, desvio `0,1341 kWh`, mediana `−0,0321 kWh`, `P5–P95 = [−0,1518, +0,2488] kWh`.
  * XGBoost: média `−0,0065 kWh`, desvio `0,1337 kWh`, mediana `−0,0087 kWh`, `P5–P95 = [−0,186, +0,208] kWh`.
  * **MLP:** média **`−0,0007 kWh`**, desvio **`0,0992 kWh`**, mediana `−0,0166 kWh`, `P5–P95 = [−0,1058, +0,1740] kWh`.

### 3.2 Alvo 2 — Queda do Estado de Carga (`delta_soc_percent`)

| Modelo | CV 5-Fold R² (388) | CV MAE | CV RMSE | **Teste R² (98)** | **Teste MAE** | **Teste RMSE** | **Teste MAPE** | **MAE / 100 km** | Negativos no Teste |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy (Média — sanidade) | — | — | — | `−0,0020` | `2,6866 %` | `3,6576 %` | `88,25%` | `100,80 %` | `0` |
| **Regressão Linear (Baseline)** | `0,7889 ± 0,0806` | `1,3667 %` | `1,8998 %` | `0,6906` | `1,3205 %` | `2,0325 %` | `38,99%` | `37,44 %` | `3` (`−0,5681`) |
| **XGBoost** | `0,8286 ± 0,0589` | `1,1596 %` | `1,7205 %` | `0,7336` | `1,1677 %` | `1,8859 %` | `30,00%` | `31,98 %` | **`0`** |
| **MLP (`(16, 8)`, `tanh`, `α=20,0`)** | **`0,8523 ± 0,0580`** | **`1,1158 %`** | **`1,5818 %`** | **`0,8000`** | **`1,1020 %`** | **`1,6342 %`** | **`28,31%`** | **`28,06 %`** | `1` (`−0,2894`) |

* **Resíduos no Teste (`98` viagens):**
  * Regressão Linear: média `+0,4174 %`, desvio `1,9892 %`, mediana `+0,3762 %`, `P5–P95 = [−2,7540, +2,3720] %` (viés sistemático).
  * XGBoost: média **`+0,0520 %`**, desvio `1,8852 %`, mediana **`+0,0246 %`**, `P5–P95 = [−2,720, +2,069] %`.
  * **MLP:** média `+0,0975 %`, desvio **`1,6313 %`**, mediana `+0,1210 %`, `P5–P95 = [−2,5151, +2,0051] %`.

### 3.3 Consenso Físico de Importância das Variáveis (Top 5 SHAP no Teste)

| Pos. | Regressão Linear (`LinearExplainer`) | XGBoost (`TreeExplainer`) | MLP (`KernelExplainer`) |
| :---: | :--- | :--- | :--- |
| **Energia 1º** | `elevation_gain_m` (`0,2660 kWh`, `+`) | `elevation_gain_m` (`0,2164 kWh`) | `distance_km` (`0,1819 kWh`) |
| **Energia 2º** | `distance_km` (`0,2034 kWh`, `+`) | `distance_km` (`0,1384 kWh`) | `elevation_gain_m` (`0,1337 kWh`) |
| **Energia 3º** | `elevation_loss_m` (`0,1600 kWh`, `−`) | `aux_hvac_energy_kwh` (`0,1023 kWh`) | `aux_hvac_energy_kwh` (`0,1245 kWh`) |
| **Energia 4º** | `aux_hvac_energy_kwh` (`0,1346 kWh`, `+`) | `duration_s` (`0,0489 kWh`) | `duration_s` (`0,0587 kWh`) |
| **Energia 5º** | `mean_oat_c` (`0,0439 kWh`, `−`) | `mean_oat_c` (`0,0325 kWh`) | `mean_oat_c` (`0,0544 kWh`) |
| **`ΔSOC` 1º** | `distance_km` (`1,8494 %`, `+`) | `distance_km` (`1,1345 %`) | `distance_km` (`1,5120 %`) |
| **`ΔSOC` 2º** | `aux_hvac_energy_kwh` (`0,8109 %`, `+`) | `aux_hvac_energy_kwh` (`0,7166 %`) | **`initial_soc_percent` (`0,7796 %`)** |
| **`ΔSOC` 3º** | `elevation_loss_m` (`0,7031 %`, `−`) | `elevation_gain_m` (`0,6203 %`) | `aux_hvac_energy_kwh` (`0,7230 %`) |
| **`ΔSOC` 4º** | `elevation_gain_m` (`0,6924 %`, `+`) | `duration_s` (`0,6125 %`) | `duration_s` (`0,6178 %`) |
| **`ΔSOC` 5º** | `initial_soc_percent` (`0,3340 %`, `−`) | `initial_soc_percent` (`0,4639 %`) | `elevation_gain_m` (`0,5773 %`) |

---

## 4. Conclusões Principais: O que os Resultados Respondem

### 4.1 É possível fazer a predição do consumo energético?
**Sim — com alta precisão para energia líquida e boa precisão operacional para queda percentual de bateria, utilizando apenas 12 atributos físicos.**
* Na **Energia Líquida**, o `MLP` alcançou `R² = 0,9616` e `MAE = 0,0738 kWh` (`73,8 Wh` por viagem; `2,18 kWh/100 km`). Para a bateria nominal de `24 kWh` do Nissan Leaf, esse erro médio equivale a aproximadamente **0,3% da capacidade total da bateria**.
* Na **queda de carga (`ΔSOC`)**, o `MLP` atingiu `R² = 0,8000` e `MAE = 1,10 p.p.`, fornecendo uma estimativa confiável mesmo sob arredondamentos do sistema de gerenciamento da bateria (BMS) e efeitos térmicos severos (`−14,45 °C` a `35,51 °C`).
* **Ressalva fundamental de aplicabilidade:** os experimentos atuais utilizam a energia auxiliar **medida** de climatização (`aux_hvac_energy_kwh`) e trajetos urbanos/periurbanos de até `21,5 km` de um único modelo de veículo elétrico. Para uso estritamente **pré-partida**, é necessária uma etapa prévia de estimativa térmica do HVAC.

### 4.2 O que a comparação entre os três algoritmos revela?
1. **A física do consumo em `kWh` tem forte componente aditivo/contínuo, mas não é puramente linear:**
   * A **Regressão Linear** já explica `92,95%` da Energia Líquida no teste (`CV = 0,9566`), demonstrando o peso somativo de distância, relevo (subidas menos recuperação regenerativa nas descidas) e HVAC. Porém, falha ao capturar interações não lineares no `ΔSOC` (`R² = 0,6906`, viés residual de `+0,417%`) e viola o limite físico em viagens curtas (`1` previsão negativa na energia e `3` no `ΔSOC`).
2. **Árvores (`XGBoost`) protegem o domínio físico e melhoram o `ΔSOC`, mas sofrem com partições em degrau em bases pequenas:**
   * O **XGBoost** superou a Linear no `ΔSOC` (`0,7336` vs. `0,6906`) e nunca previu valores negativos (`0` em ambos os alvos). Entretanto, em uma amostra de `388` viagens com alvos contínuos, árvores produzem patamares discretos e não extrapolam acima do teto das folhas vistas no treino (por isso ficou em `CV R² = 0,9192` na Energia contra `0,9566` da Linear e `0,9681` do MLP).
3. **O `MLP` enxuto e regularizado uniu o melhor dos dois mundos:**
   * Combinou a **continuidade suave** (superando o teto de degraus das árvores na Energia Líquida: `MAE` `~25%` menor que o XGBoost, `0,0738` vs. `0,0985 kWh`) com a **modelagem não linear** das duas camadas ocultas `(16, 8)` (superando a Linear e o XGBoost no `ΔSOC`: `R² = 0,8000`, `+6,64 p.p.` sobre o XGBoost e `+10,94 p.p.` sobre a Linear).
4. **Lição metodológica sobre redes neurais em dados tabulares pequenos:**
   * O contraste entre a Etapa 1 (`ΔSOC CV R² = 0,6451`, `5` falhas de convergência, `25` negativos) e a Etapa 2/3 (`CV R² = 0,8523`, `Holdout R² = 0,8000`, `0` avisos) comprova que redes neurais em ~400 amostras **exigem** padronização em `Pipeline`, arquitetura compacta, otimizador adequado (`lbfgs`) e regularização L2 calibrada à escala do alvo na CV.
5. **Consenso físico independente do algoritmo:**
   * Os três modelos concordam que **distância, ganho acumulado em aclives (SRTM) e climatização (HVAC) / temperatura externa** governam o consumo líquido, enquanto o **estado de carga inicial (`initial_soc_percent`)** assume papel central na explicação do `ΔSOC` (2º lugar no SHAP do MLP com `0,7796 %`), refletindo a eletroquímica da bateria e a limitação de frenagem regenerativa próxima de 100% de carga.

---

## 5. Qual Algoritmo Escolher para um Sistema Futuro em Tempo Real?

### 5.1 Recomendação Principal: `MLP Regressor` com Trava Física `max(0, y_pred)`
Entre os modelos testados, o **`MLP`** é o melhor candidato para embarcar em uma aplicação futura de estimativa de consumo e autonomia:
1. **Maior acurácia nos dois alvos** (`73,8 Wh` de erro médio na energia e `1,10 p.p.` no `ΔSOC`).
2. **Custo computacional mínimo na inferência em tempo real:** depois de treinado, o modelo `(16, 8)` resume-se a apenas **353 pesos** (três matrizes pequenas: `12×16`, `16×8`, `8×1`) somados a `24` constantes do `StandardScaler` (`12` médias e `12` desvios). A inferência ocorre em **microssegundos** em CPU comum, smartphone ou módulo embarcado, sem GPU.
3. **Respostas suaves no painel:** por gerar uma função contínua, pequenas alterações na rota ou na temperatura atualizam a autonomia estimada sem "saltos" abruptos entre faixas de árvores.
4. **Salvaguarda simples de produção:** aplicar `max(0.0, y_pred)` na camada de saída elimina qualquer chance de previsão residual negativa em deslocamentos muito curtos (como o único caso de `−0,2894 %` no `ΔSOC`).

### 5.2 Papéis Complementares dos Outros Dois Algoritmos
* **`XGBoost`:** continua sendo uma excelente alternativa caso o sistema futuro precise lidar diretamente com **sensores falhando (`NaN`)**, incorporar **muitas variáveis categóricas heterogêneas** (múltiplos modelos/marcas de veículos, tipos de via/pneu) ou priorizar pipelines com menor sensibilidade à calibração de escala e regularização.
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
  * `src/xgboost_consumption_model.py` (Dummy, Regressão Linear Baseline + CV + SHAP, XGBoost + CV + SHAP e figuras `25_`–`27_`, `35_`–`38_`).
  * `src/mlp_consumption_model.py` (MLP em 3 etapas + refinamento `2b` via `--stage 1 | 2 | 2b | 3` e figuras `39_`–`41_`).
* **Relatórios numéricos (`outputs/results/trips/`):**
  * `linear_regression_baseline_evaluation.json`
  * `xgboost_dual_model_evaluation.json`, `xgboost_dual_model_evaluation.csv`, `xgboost_shap_rankings.csv`
  * `mlp_etapa1_diagnostic.json`, `mlp_etapa2_cv_tuning.json`, `mlp_etapa2_cv_tuning.csv`, `mlp_etapa2b_refinement.json`, `mlp_etapa2b_refinement.csv`
  * `mlp_dual_model_evaluation.json`, `mlp_shap_rankings.csv`
  * `phase2_three_models_comparison.csv`
* **Modelos serializados (`outputs/models/`):**
  * `xgboost_energy_net_model.joblib`, `xgboost_delta_soc_model.joblib`
  * `mlp_energy_net_model.joblib`, `mlp_delta_soc_model.joblib`
* **Figuras geradas (`outputs/figures/trips/`):**
  * XGBoost: `25_xgboost_real_vs_pred_energy.png`, `26_xgboost_residuals_distribution.png`, `27_xgboost_feature_importance_shap.png`
  * Regressão Linear: `36_linear_regression_real_vs_pred_energy.png`, `37_linear_regression_residuals_distribution.png`, `38_linear_regression_feature_importance_shap.png`
  * MLP: `39_mlp_real_vs_pred_energy.png`, `40_mlp_residuals_distribution.png`, `41_mlp_feature_importance_shap.png`
  * Comparativo inicial (Dummy × Linear × XGBoost): `35_comparacao_modelos_baselines_xgboost.png`
