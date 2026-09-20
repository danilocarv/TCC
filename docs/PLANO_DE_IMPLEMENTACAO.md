# Plano de Implementação Modular - TCC: Predição de Consumo Energético e Perfis de Condução em Veículos Elétricos

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** Instituto Federal de Educação, Ciência e Tecnologia do Sul de Minas Gerais (IFSULDEMINAS) - Campus Poços de Caldas  
**Base de Dados:** Vehicle Energy Dataset (VED)  
**Veículos de Estudo:** Nissan Leaf 2013 (100% Elétricos - IDs 10, 455, 541)  
**Data de Criação:** Setembro/2026  

---

## 1. Visão Geral e Estratégia de Desenvolvimento

O projeto foi estruturado em **duas grandes fases independentes e sequenciais**, garantindo validação experimental antes do avanço para a etapa seguinte:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ FASE 1: CLASSIFICAÇÃO DE PERFIS DE CONDUÇÃO (FOCO IMEDIATO)                  ║
║  • 1.1 Ingestão e extração exclusiva dos dados de veículos 100% elétricos    ║
║  • 1.2 Engenharia de atributos comportamentais (cinemática e eventos bruscos)║
║  • 1.3 Clusterização com K-Means (análise de cotovelo, silhueta e centróides)║
║  • 1.4 Interpretação dos perfis (Econômico, Moderado e Agressivo)            ║
║  • 1.5 Módulo de teste interativo (classificação em tempo real)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                       │
                                       ▼ [Validação e Aprovação pelo Danilo]
╔══════════════════════════════════════════════════════════════════════════════╗
║ FASE 2: PREDIÇÃO DE CONSUMO ENERGÉTICO (ETAPA POSTERIOR)                     ║
║  • 2.1 Integração de GPS e cálculo de altitude/declividade da via            ║
║  • 2.2 Modelagem supervisionada (Regressão Linear, Random Forest, Redes MLP) ║
║  • 2.3 Avaliação do impacto da inclusão dos perfis de condução no consumo    ║
║  • 2.4 Preditor de consumo para trajeto do Ponto A ao Ponto B                ║
║  • 2.5 Dashboard interativo completo (Streamlit) e gráficos para o artigo    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Estrutura do Repositório do Projeto

```
TCC/
├── docs/                                  # Documentação, proposta da pré-banca e planos
│   ├── Pré-banca de TCC - Danilo Carvalho de Oliveira.pdf
│   ├── PLANO_DE_IMPLEMENTACAO.md          # Este documento com o roteiro completo
│   └── REGISTRO_DE_DESENVOLVIMENTO.md     # Diário de bordo, mapeamento e histórico de código
├── data/
│   ├── raw/                               # Dados brutos extraídos (EVs puros: 476.308 linhas)
│   └── processed/                         # Dados com features calculadas e clusters rotulados
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configurações globais, hiperparâmetros e caminhos
│   ├── data_loader.py                     # Leitura e consolidação dos dados do VED
│   ├── preprocessing.py                   # Limpeza, tratamento de nulos e ruídos
│   ├── feature_engineering.py             # Atributos cinemáticos, janelamento e eventos
│   ├── clustering.py                      # K-Means, métricas (Silhouette, Elbow), rotulagem
│   ├── regression.py                      # Modelos preditivos de consumo (Fase 2)
│   ├── evaluation.py                      # Métricas MAE, RMSE, R² (Fase 2)
│   └── visualization.py                   # Plots acadêmicos em alta resolução (300 DPI)
├── notebooks/                             # Notebooks Jupyter para exploração didática
├── app/                                   # Aplicações interativas (Streamlit)
├── outputs/
│   ├── figures/                           # Imagens e gráficos gerados para o TCC
│   ├── models/                            # Modelos treinados serializados (.joblib)
│   └── results/                           # Tabelas com métricas consolidadas
├── VED-master/                            # Base de dados original completa
├── requirements.txt                       # Dependências Python do projeto
└── README.md                              # Guia rápido de introdução e execução
```

---

## 3. Detalhamento Técnico da FASE 1 (Perfis de Condução)

### Etapa 1.1: Extração e Consolidação dos Dados EV
* **Objetivo:** Filtrar os 54 arquivos semanais do VED mantendo apenas os veículos puramente elétricos:
  * `VehId` 10, 455 e 541 (Nissan Leaf 2013, bateria de 24 kWh).
  * Volume identificado: **476.308 registros** a 1 Hz distribuídos em **491 viagens reais**.
* **Variáveis extraídas:** `DayNum`, `VehId`, `Trip`, `Timestamp(ms)`, `Latitude[deg]`, `Longitude[deg]`, `Vehicle Speed[km/h]`, `HV Battery Current[A]`, `HV Battery Voltage[V]`, `HV Battery SOC[%]`, `OAT[DegC]`, `Air Conditioning Power[Watts]`, `Heater Power[Watts]`.
* **Saída:** Arquivo persistido em `data/raw/ev_telemetry.parquet` (para carregamento ultrarrápido) e `data/raw/ev_telemetry.csv`.

### Etapa 1.2: Engenharia de Atributos Comportamentais
* **Variáveis Cinemáticas Ponto a Ponto (1 Hz):**
  * Delta de tempo: $\Delta t = (t_k - t_{k-1}) / 1000$ (s).
  * Aceleração longitudinal: $a = \frac{v_k - v_{k-1}}{\Delta t}$ ($m/s^2$).
  * Variação de aceleração (*jerk*): $j = \frac{a_k - a_{k-1}}{\Delta t}$ ($m/s^3$).
* **Janelamento / Segmentação Temporal de Condução (conforme literatura Mobini Seraji et al., 2025):**
  * Trechos de condução de 60 a 180 segundos.
  * Média da velocidade ($\bar{v}$) e velocidade máxima ($v_{max}$).
  * Desvio padrão da velocidade ($\sigma_v$) — indicador de instabilidade do condutor.
  * Aceleração positiva média ($\bar{a}_{pos}$) e aceleração máxima ($a_{max}$).
  * Frequência e severidade de frenagens bruscas ($a < -2.0 \text{ m/s}^2$).
  * Taxa de tempo em ociosidade / veículo parado (*idle ratio*).
* **Saída:** Dataset agregado por trecho em `data/processed/driving_behavior_features.csv`.

### Etapa 1.3: Clusterização Não Supervisionada (K-Means)
* **Normalização:** `StandardScaler` / `RobustScaler` para remover viés de escala entre variáveis.
* **Determinação do Número Ótimo de Clusters ($k$):**
  * Curva da Inércia (Método do Cotovelo / *Elbow Method*).
  * Coeficiente de Silhueta (*Silhouette Score*) para $k \in [2, 6]$.
  * Índice Davies-Bouldin e Calinski-Harabasz.
* **Rotulagem Semântica dos Grupos ($k=3$):**
  1. **Econômico / Suave:** Condução calma, baixa variância de velocidade, acelerações brandas, frenagens suaves.
  2. **Moderado / Normal:** Comportamento equilibrado no trânsito padrão.
  3. **Agressivo / Esportivo:** Acelerações repentinas, frenagens bruscas recorrentes, alta oscilação de velocidade.
* **Visualização:** Gráficos de dispersão em 2D e 3D após redução por PCA e t-SNE.

### Etapa 1.4: Módulo Interativo de Teste e Classificação
* Desenvolvimento de script interativo (`test_profile.py` / mini-interface):
  * Permite ao usuário escolher qualquer trajeto real da base para inspecionar o perfil detectado.
  * Permite ao usuário **digitar valores hipotéticos ou novos** (ex.: velocidade média, número de frenagens bruscas, pico de aceleração) e a IA responde instantaneamente:
    * O Perfil de Condução atribuído (*Econômico*, *Moderado* ou *Agressivo*).
    * O grau de pertinência/distância em relação a cada centróide.
    * Um comparativo visual do perfil inserido em relação à média dos outros perfis.

---

## 4. Detalhamento Técnico da FASE 2 (Predição de Consumo Energético)
*(A ser iniciada e detalhada formalmente após a validação da Fase 1)*
* Cálculo de elevação e declividade da via a partir de Latitude/Longitude.
* Potência instantânea da bateria: $P = V \times I$ (kW) e consumo acumulado (kWh).
* Treinamento dos modelos: **Regressão Linear**, **Random Forest** e **Redes Neurais Artificiais (MLP)**.
* Experimento comparativo:
  * **Cenário A:** Sem o atributo de perfil de condução.
  * **Cenário B:** Com o atributo de perfil de condução integrado.
* Avaliação de métricas: $MAE$, $RMSE$, $R^2$.
* Simulador de percurso do Ponto A ao Ponto B com base no perfil.
* Dashboard interativo final em Streamlit e figuras de 300 DPI para o artigo.

