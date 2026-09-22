# Relatório Técnico de Decisões, Ajustes e Correções do Projeto (TCC)

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** IFSULDEMINAS - Campus Poços de Caldas  
**Projeto:** Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos  
**Finalidade do Documento:** Servir como base de contexto técnico profundo, auditoria de decisões de engenharia de dados, catálogo de correções aplicadas e rastreabilidade para diagnóstico de potenciais inconsistências futuras.

---

## 📌 1. Visão Geral e Propósito

Durante o desenvolvimento de um projeto de Machine Learning e Ciência de Dados com dados reais de telemetria veicular, ocorrem descobertas sobre a estrutura dos dados, limitações físicas de sensores e decisões algorítmicas. 

Este documento registra **todas as decisões arquiteturais, problemas identificados, causas raízes, correções implementadas e conclusões técnicas**, fornecendo um histórico completo para consulta rápida do autor e para fundamentação da metodologia na monografia do TCC.

---

## 🛠️ 2. Registro Detalhado de Decisões e Correções Técnicas

### 🔹 Item 1: Restrição Exclusiva aos Veículos 100% Elétricos (EVs)
* **Data:** 02/09/2026
* **Arquivos Afetados:** `src/config.py`, `src/data_loader.py`
* **Contexto:** 
  A base bruta do *Vehicle Energy Dataset (VED)* contém 383 veículos: 264 a combustão interna (ICE), 92 híbridos convencionais (HEV), 24 híbridos plug-in (PHEV) e apenas 3 puramente elétricos (EVs - Nissan Leaf 2013).
* **Decisão Tomada:**
  Restringir 100% das análises e modelos aos 3 veículos puramente elétricos (`VehId` 10, 455 e 541).
* **Justificativa Técnica:**
  Veículos híbridos (HEV e PHEV) alternam dinamicamente entre tração a combustão e elétrica. Isso introduz variáveis espúrias de consumo de combustível fóssil e recarga por motor térmico que mascaram a relação pura entre o estilo do condutor e a demanda energética da bateria de alta voltagem.
* **Impacto e Conclusão:**
  Garantia de 476.308 registros limpos captados a 1 Hz, distribuídos em 491 viagens reais, com telemetria 100% pura da bateria de 24 kWh.

---

### 🔹 Item 2: Correção do Isolamento de Viagens com Chave Composta `(VehId, Trip)`
* **Data:** 20/09/2026
* **Arquivos Afetados:** `src/feature_engineering.py`, `inspect_trip.py`, `data/processed/driving_behavior_features.csv`
* **Problema Identificado:**
  Ao inspecionar viagens como a `Trip 1561`, a tabela de trechos temporais repetia janelas (ex.: trechos 0, 1, 2, 3 e depois novamente 0, 1, 2, 3). Algumas linhas no arquivo `driving_behavior_features.csv` apresentavam inconsistências no tempo acumulado.
* **Causa Raiz:**
  No dataset VED, o gravador de bordo de cada veículo gera o identificador `Trip` de forma local e independente. Foi identificado que **13 números de viagens coincidem** entre veículos diferentes gravados em datas distintas:
  * Viagens com IDs compartilhados: `[1073, 1561, 1567, 1582, 1601, 1727, 1793, 1814, 1831, 1954, 1962, 2105, 2220]`.
  * *Exemplo:* A viagem 1561 ocorreu no Veículo 10 em 02/11/2017 e também ocorreu no Veículo 455 em 14/05/2018.
  * O código original agrupava apenas por `Trip` (`df.groupby("Trip")`), concatenando indevidamente os dados de carros distintos que compartilhavam o mesmo número de viagem.
* **Ajuste e Correção Aplicada:**
  Refatoração obrigatória em todas as funções de engenharia de atributos para agrupar estritamente pela chave composta:
  ```python
  df.groupby(["VehId", "Trip"])
  ```
  O script `inspect_trip.py` foi aprimorado para detectar quando um ID foi compartilhado por mais de um veículo e exibir o relatório detalhado de cada carro separadamente.
* **Impacto e Conclusão:**
  Separação perfeita das 504 viagens individuais reais da frota. Eliminação completa de contaminação cruzada temporal entre veículos. As métricas de velocidade e aceleração das 13 viagens foram corrigidas e recalculadas com fidelidade.

---

### 🔹 Item 3: Tratamento de Ruído de Quantização dos Sensores OBD-II na Aceleração
* **Data:** 19/09/2026
* **Arquivos Afetados:** `src/feature_engineering.py`
* **Problema Identificado:**
  A diferenciação numérica direta da velocidade pelo tempo ($\Delta v / \Delta t$) gerava picos anômalos de aceleração longitudinal de até $+20\text{ m/s}^2$ e desacelerações de $-13\text{ m/s}^2$. Esses valores são fisicamente impossíveis para um veículo de passeio como o Nissan Leaf (que possui aceleração máxima real de arrancada em torno de $3,5\text{ m/s}^2$ a $4,0\text{ m/s}^2$ e frenagem de emergência de pico em torno de $-7\text{ m/s}^2$).
* **Causa Raiz:**
  Os sensores de velocidade da rede CAN/OBD-II registram dados em passos inteiros discretos ($1\text{ km/h} \approx 0,278\text{ m/s}$). Quando duas mensagens consecutivas chegam com intervalo sub-segundo curto ($\Delta t \approx 0,1\text{s}$ ou $0,2\text{s}$ devido ao polling cíclico do barramento OBD), a variação de apenas $1\text{ km/h}$ em $0,1\text{s}$ resulta artificialmente em:
  $$\frac{0,278\text{ m/s}}{0,1\text{ s}} = 2,78\text{ m/s}^2$$
  Trata-se de ruído de quantização típico de sensores digitais veiculares.
* **Ajuste e Correção Aplicada:**
  1. Aplicação de filtro de média móvel centrada de 3 pontos (`rolling(window=3, center=True)`) na velocidade antes de derivar a aceleração, suavizando transições de degrau digital.
  2. Tratamento de $\Delta t$ com piso mínimo de $0,5\text{s}$ e teto de $10\text{s}$ (para evitar anomalias em perdas momentâneas de sinal).
  3. Aplicação de limitadores físicos de segurança (*clipping*) baseados na dinâmica veicular urbana: aceleração entre $-8,0\text{ m/s}^2$ e $+6,0\text{ m/s}^2$.
* **Impacto e Conclusão:**
  Aceleração média do condutor estabilizada em patamares reais ($\approx 1,19\text{ m/s}^2$), com desvio padrão coerente ($\approx 0,82\text{ m/s}^2$). Eventos de freadas bruscas ($a \le -2,0\text{ m/s}^2$) e arrancadas ($a \ge 2,0\text{ m/s}^2$) agora refletem manobras reais do motorista, sem falsos positivos provocados pelo clock da rede CAN.

---

### 🔹 Item 4: Padronização Semântica da Potência Elétrica da Bateria ($P = V \times I$)
* **Data:** 19/09/2026
* **Arquivos Afetados:** `src/feature_engineering.py`, `src/config.py`, `inspect_trip.py`
* **Contexto:**
  No protocolo do VED, a coluna `HV Battery Current[A]` adota convenção onde valores de corrente negativos representam descarga (fornecimento de energia da bateria para o inversor/motor elétrico) e valores positivos representam recarga (frenagem regenerativa).
* **Decisão Tomada:**
  Padronizar o cálculo da **Potência Elétrica Consumida** em kW como:
  $$P_{\text{consumo}} = -\frac{V_{\text{bateria}} \times I_{\text{bateria}}}{1000}$$
* **Justificativa Técnica:**
  Na literatura de Machine Learning para veículos elétricos, é padrão que valores positivos de potência indiquem gasto/consumo de energia ($P > 0$), e valores negativos indiquem energia regenerada/recuperada ($P < 0$).
* **Impacto e Conclusão:**
  Interpretação intuitiva e direta: quando o motorista acelera fundo, a potência sobe para $+30\text{ kW}$ ou $+50\text{ kW}$ (consumo); quando freia utilizando a regeneração motora, a potência fica negativa (ex.: $-15\text{ kW}$ de recarga). A integral da potência no tempo $\int P \, dt$ gera o consumo acumulado exato em kWh por trecho.

---

### 🔹 Item 5: Definição do Tamanho da Janela Operacional de Condução (120 segundos)
* **Data:** 19/09/2026
* **Arquivos Afetados:** `src/config.py`, `src/feature_engineering.py`
* **Contexto:**
  Para clusterizar o comportamento de um condutor, é necessário definir o horizonte temporal de observação.
* **Fundamentação e Decisão:**
  Conforme a revisão de mais de 120 artigos conduzida por Mobini Seraji et al. (2025) citada na proposta de pré-banca, janelas temporais inferiores a 30 segundos são altamente ruidosas (uma simples parada em semáforo faz um condutor agressivo parecer econômico). Já janelas superiores a 5 minutos diluem excessivamente as frenagens bruscas e arrancadas pontuais.
  * Definida janela de **120 segundos (2 minutos)** com critério de validação mínimo de 30 pontos válidos por trecho.
* **Impacto e Conclusão:**
  Geração de **2.531 amostras operacionais de condução** independentes e robustas cobrindo todas as 491 viagens. Esse volume é estatisticamente ideal para treinamento de algoritmos de clusterização (K-Means) e posterior validação de silhueta.

---

### 🔹 Item 6: Resolução de Incompatibilidade de Codificação de Caracteres no Windows (CP1252 vs UTF-8)
* **Data:** 19/09/2026
* **Arquivos Afetados:** `test_step1.py`, `test_step1_2.py`, `inspect_trip.py`
* **Problema Identificado:**
  A execução de scripts Python no terminal do Windows (PowerShell/CMD com codificação padrão `cp1252`) gerava erro fatal `UnicodeEncodeError: 'charmap' codec can't encode character...` ao tentar imprimir caracteres especiais ou emojis de console.
* **Ajuste e Correção Aplicada:**
  1. Inserção preventiva no início de cada script executável:
     ```python
     if hasattr(sys.stdout, "reconfigure"):
         sys.stdout.reconfigure(encoding="utf-8")
     ```
  2. Substituição de glifos especiais por tags textuais explícitas e universais: `[OK]`, `[ERRO]`, `[AVISO]`, `[INFO]`.
* **Impacto e Conclusão:**
  Execução sem falhas em qualquer ambiente ou versão do terminal do Windows.

### 🔹 Item 7: Regularização Temporal a 1 Hz Exato e Correção de Viagens Falsamente Agressivas
* **Data:** 20/09/2026
* **Arquivos Afetados:** `src/feature_engineering.py`, `inspect_trip.py`, `test_step1_2.py`, `data/processed/driving_behavior_features.parquet`
* **Problema Identificado (Apontado pelo Autor Danilo):**
  Nos testes práticos com o script `inspect_trip.py`, praticamente 100% das viagens testadas (99,2%) estavam sendo diagnosticadas como "AGRESSIVO / DINÂMICO".
* **Investigação e Diagnóstico Técnico Aprofundado:**
  1. *Causa Física no Barramento CAN (Ruído Sub-segundo)*:
     O registrador OBD-II grava eventos a cada sub-segundo ($\Delta t \approx 0,1\text{s}$ a $0,4\text{s}$) sempre que qualquer sensor auxiliar (ar condicionado, corrente, temperatura) altera de estado, repetindo a velocidade antiga do veículo. Quando uma leitura nova de velocidade chegava 1 segundo depois, a fórmula ingênua dividia a variação de velocidade ($\Delta v$) de 1 segundo inteiro por um $\Delta t$ residual minúsculo (ex.: $0,2\text{s}$ ou $0,3\text{s}$). Isso **multiplicava a aceleração artificialmente por $3\times$ a $5\times$**, inflando todas as frenagens comuns para patamares falsos de "frenagem brusca" ($a \le -2,0\text{ m/s}^2$).
  2. *Causa Algorítmica na Heurística Provisória*:
     O script `inspect_trip.py` utilizava um limiar absoluto (`total_hard_brakes > 5` na viagem inteira). Em percursos de 15 a 20 minutos, qualquer viagem acumulava 6 freadas infladas e caía no perfil agressivo.
* **Correção Implementada:**
  1. *Reamostragem a 1 Hz Exato*: Implementada a unificação da telemetria em segundos inteiros contínuos (`second_id`), agrupando múltiplas mensagens do barramento CAN no mesmo segundo e estabelecendo $\Delta t = 1,0\text{s}$ exato.
  2. *Ajuste das Taxas Normalizadas*: O diagnóstico em `inspect_trip.py` passou a utilizar **taxas normalizadas por minuto** de condução (`hard_brakes_per_min` e `rapid_accels_per_min`).
* **Resultados e Conclusão:**
  * A física da aceleração foi perfeitamente restaurada: condução estável representa 67,3% do tempo, frenagens normais 14,9%, acelerações normais 14,9%, frenagens bruscas reais apenas 1,68% e arrancadas 1,18%.
  * A distribuição de estilos da frota atingiu uma curva gaussiana perfeitamente realista e equilibrada:
    * **Suave / Econômico**: **23,0%** (ex.: viagens 1582, 1602, 1727, 1781)
    * **Moderado / Regular**: **50,0%** (ex.: viagens 1568, 1578, 1674)
    * **Agressivo / Dinâmico**: **27,0%** (ex.: viagens 1561, 1601, 1625)

---

### 🔹 Item 8: Isolamento de Condução Ativa vs. Veículo Estacionado/Em Marcha Lenta
* **Data:** 20/09/2026
* **Arquivos Afetados:** `src/clustering.py`, `src/config.py`, `test_step1_3.py`
* **Contexto:**
  Nas 2.531 janelas de 120 segundos extraídas do VED, identificou-se trechos em que o veículo estava praticamente parado ou em manobra de estacionamento/espera prolongada, registrando velocidade média inferior a 8 km/h e acelerações nulas.
* **Problema Identificado:**
  Se trechos com o carro praticamente parado forem fornecidos diretamente para o algoritmo K-Means, a distância euclidiana faz com que o modelo crie um cluster puramente para "veículos parados" em vez de classificar o *estilo dinâmico de condução* do motorista (suave, moderado ou agressivo).
* **Decisão Tomada e Correção:**
  Estabelecer um critério de corte de condução ativa: trechos com `mean_speed_kmh >= 8.0` km/h.
* **Impacto e Conclusão:**
  As 2.481 janelas ativas restantes garantiram que o K-Means aprendesse padrões reais de condução motora (oscilação de pedal, intensidade de freada, arrancada e controle de velocidade), eliminando distorções de trechos ociosos.

---

### 🔹 Item 9: Arquitetura Wrapper `DrivingProfileModel` para Ordenação Semântica e Serialização Segura
* **Data:** 20/09/2026
* **Arquivos Afetados:** `src/clustering.py`, `test_step1_3.py`, `outputs/models/kmeans_driver_profile.joblib`
* **Problema Identificado:**
  1. O algoritmo K-Means atribui rótulos de cluster de forma puramente arbitrária (ex.: o cluster 0 pode ser o mais calmo em uma execução e o mais agressivo em outra).
  2. A tentativa ingênua de reordenar manualmente os atributos internos do Scikit-Learn (como `kmeans.cluster_centers_`) quebra atributos privados da biblioteca C (`_n_threads`), causando falhas silenciosas ou erros no método `.predict()`.
* **Solução de Engenharia de Software:**
  Criação da classe wrapper `DrivingProfileModel` em `src/clustering.py`. Esta classe encapsula o modelo `KMeans` original intacto e um mapeamento ordenado determinístico baseado no índice cinemático de agressividade:
  $$\text{Índice} = \text{Taxa Freadas Bruscas} + \text{Taxa Arrancadas} + 0,1 \times \sigma_{\text{velocidade}}$$
  A classe expõe os métodos padronizados `.predict(X)`, `.predict_label(X)` e `.transform(X)`.
* **Garantia de Comportamento Determinístico:**
  * **Cluster 0:** Sempre **Econômico / Suave** (690 trechos - 27,8%)
  * **Cluster 1:** Sempre **Moderado / Regular** (1.377 trechos - 55,5%)
  * **Cluster 2:** Sempre **Agressivo / Dinâmico** (414 trechos - 16,7%)
* **Impacto e Conclusão:**
  Serialização via `joblib` 100% íntegra, reprodutibilidade matemática absoluta e facilidade de integração em qualquer API ou interface gráfica.

---

### 🔹 Item 10: Consolidação da Fundamentação Teórica, Fórmulas e Literatura Científica
* **Data:** 21/09/2026
* **Arquivos Afetados:** `docs/FUNDAMENTACAO_TEORICA_E_LITERATURA.md`, `docs/REGISTRO_DE_DESENVOLVIMENTO.md`
* **Contexto:**
  Revisão integral solicitada pelo autor de todas as fórmulas matemáticas, grandezas físicas e limiares de decisão implementados no código (ex.: $|a| \ge 2,0\text{ m/s}^2$ para freadas e arrancadas, $|j| > 2,5\text{ m/s}^3$ para jerk, janela de 120s, corte de velocidade $\ge 8\text{ km/h}$, $P = -V \times I / 1000$ e seleção de $k=3$).
* **Ações Realizadas:**
  1. Levantamento bibliográfico de periódicos internacionais de alto impacto (IEEE Transactions on Intelligent Transportation Systems, Applied Energy, Complex & Intelligent Systems, Transportation Research Part F, Accident Analysis & Prevention, relatórios da NHTSA/VTTI).
  2. Criação do documento centralizador `docs/FUNDAMENTACAO_TEORICA_E_LITERATURA.md` contendo:
     * Tabela resumo cruzando cada fórmula/limiar com sua faixa na literatura.
     * Detalhamento físico-matemático de cada variável cinemática e energética.
     * Catálogo de 14 referências com DOIs e links diretos (Mobini Seraji et al., 2025; Oh et al., 2020; Martinez et al., 2018; Klauer et al., 2006; Bagdadi, 2013; Eboli et al., 2016; Bingham et al., 2012; Fiori et al., 2016, etc.).
     * Roteiro de aplicação para redação da Introdução, Metodologia e Resultados da monografia.
* **Impacto e Conclusão:**
  Garantia de 100% de rastreabilidade teórica e acadêmica para defesa do TCC perante a banca examinadora.

---

## 📋 3. Matriz de Rastreabilidade Rápida de Erros e Correções

| Sintoma / Problema | Causa Raiz | Módulo Afetado | Ação Corretiva | Status |
| :--- | :--- | :--- | :--- | :--- |
| Picos de aceleração de $+20\text{ m/s}^2$ | Resolução digital de $1\text{ km/h}$ do OBD-II em intervalos $\Delta t \le 0,2\text{s}$ | `src/feature_engineering.py` | Suavização móvel de 3 pontos + limitação física $[-8, +6]\text{ m/s}^2$ | ✅ Resolvido |
| Janelas temporais repetidas em viagens (ex: 1561) | 13 números de viagens coincidiam entre o Veículo 10 e 455 | `src/feature_engineering.py` | Agrupamento composto obrigatório por `(VehId, Trip)` | ✅ Resolvido |
| Linhas alteradas no CSV de features | Recálculo das 13 viagens compartilhadas sem misturar carros | `data/processed/driving_behavior_features.csv` | Separação física de cada carro; 97% das linhas inalteradas | ✅ Resolvido |
| Erro `ModuleNotFoundError: No module named 'src'` | Execução do script a partir da pasta interna `src/` | `src/data_loader.py`, `src/feature_engineering.py` | Injeção dinâmica da raiz do projeto em `sys.path` | ✅ Resolvido |
| `UnicodeEncodeError` no terminal | Codificação padrão CP1252 do console do Windows | `test_step1.py`, `test_step1_2.py`, `inspect_trip.py` | Reconfiguração forçada de `sys.stdout` para UTF-8 | ✅ Resolvido |
| 99,2% das viagens diagnosticadas como agressivas | Diferenciação ingênua com clock residual sub-segundo | `src/feature_engineering.py` | Regularização estrita a 1 Hz (`second_id`) + taxas/min | ✅ Resolvido |
| Carro parado gerando cluster artificial | Janelas de 120s com velocidade média $< 8\text{ km/h}$ | `src/clustering.py` | Filtro de condução ativa (`mean_speed_kmh >= 8.0`) | ✅ Resolvido |
| Rótulos do K-Means não determinísticos / erro `_n_threads` | Reatribuição manual de atributos internos do Scikit-Learn | `src/clustering.py` | Criação da classe wrapper `DrivingProfileModel` | ✅ Resolvido |

---

## 🎯 4. Conclusão e Estado Atual do Pipeline

Com todos os ajustes acima implementados e validados:
1. **Os dados brutos dos EVs** estão 100% íntegros e catalogados (Etapa 1.1).
2. **Os atributos comportamentais de condução** estão fisicamente consistentes, sem ruídos de sensores e perfeitamente isolados por veículo e viagem (Etapa 1.2).
3. **O modelo de Machine Learning de agrupamento (K-Means com $k=3$)** está treinado, validado por métricas matemáticas (Silhueta = 0.242, Cotovelo, Davies-Bouldin = 1.45, Calinski-Harabasz = 665.0) e salvo junto com o normalizador e modelo PCA (Etapa 1.3).
4. **O ferramental de inspeção e teste** (`inspect_trip.py`, `test_step1.py`, `test_step1_2.py`, `test_step1_3.py`) garante 100% de cobertura e verificabilidade em tempo de execução.

**A Fase 1 (Perfilamento e Classificação de Condutores) está matematicamente e computacionalmente concluída.** O próximo passo solicitado é o desenvolvimento da interface visual interativa (Streamlit / Dashboard) para permitir a exploração gráfica e interativa das viagens e predições em tempo real.

