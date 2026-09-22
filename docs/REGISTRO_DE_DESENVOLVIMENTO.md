# Diário de Bordo e Mapeamento do Projeto (TCC)

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Projeto:** Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos  
**Status Atual:** Fase 1 - Etapa 1.3 concluída com sucesso (K-Means treinado, validado e artefatos salvos)  

---

## 🗺️ Mapa de Arquivos e Componentes do Projeto

Abaixo encontra-se o inventário atualizado de todos os componentes do repositório:

| Caminho / Arquivo | Descrição | Status |
| :--- | :--- | :--- |
| `docs/Pré-banca de TCC - Danilo Carvalho de Oliveira.pdf` | Documento original da proposta de TCC apresentada à banca | ✅ Concluído |
| `docs/PLANO_DE_IMPLEMENTACAO.md` | Plano mestre detalhado contendo a arquitetura e divisão das fases | ✅ Concluído |
| `docs/REGISTRO_DE_DESENVOLVIMENTO.md` | Este diário de bordo com o histórico e mapeamento do projeto | ✅ Ativo |
| `docs/HISTORICO_TECNICO_E_AJUSTES.md` | Relatório técnico detalhado com histórico de decisões, causas-raízes e correções | ✅ Ativo |
| `docs/FUNDAMENTACAO_TEORICA_E_LITERATURA.md` | Catálogo de fundamentação teórica, fórmulas, limiares e referências com DOIs/links | ✅ Concluído |
| `src/__init__.py` | Arquivo de inicialização do pacote principal | ✅ Concluído |
| `src/config.py` | Configurações globais, caminhos, constantes e metadados dos EVs | ✅ Concluído |
| `src/data_loader.py` | Pipeline de extração e consolidação dos dados dos veículos elétricos | ✅ Concluído |
| `src/feature_engineering.py` | Cálculo de cinemática (aceleração, jerk) e janelamento comportamental | ✅ Concluído |
| `test_step1.py` | Script de teste e validação automatizada da integridade dos dados extraídos | ✅ Concluído |
| `test_step1_2.py` | Script de teste e validação dos atributos comportamentais calculados | ✅ Concluído |
| `inspect_trip.py` | Inspecionador interativo para consultar métricas de qualquer viagem real do dataset | ✅ Concluído |
| `data/raw/ev_telemetry.parquet` | Base consolidada dos EVs puros em Parquet (3.72 MB) | ✅ Concluído |
| `data/raw/ev_telemetry.csv` | Cópia direta em CSV para inspeção tabular (41.69 MB) | ✅ Concluído |
| `data/raw/ev_dataset_metadata.json` | Metadados estatísticos e contagem de nulos do dataset extraído | ✅ Concluído |
| `data/processed/driving_behavior_features.parquet` | Base com os atributos comportamentais das 2.481 janelas ativas de condução (121 KB) | ✅ Concluído |
| `data/processed/driving_behavior_features.csv` | Tabela dos atributos comportamentais para abertura no Excel/VS Code (230 KB) | ✅ Concluído |
| `data/processed/driving_behavior_metadata.json` | Metadados descritivos das variáveis comportamentais | ✅ Concluído |
| `src/clustering.py` | Pipeline de clusterização K-Means, avaliação de métricas e ordenação semântica de perfis | ✅ Concluído |
| `test_step1_3.py` | Script de teste e validação da clusterização, métricas e predição com novos perfis | ✅ Concluído |
| `data/processed/driving_behavior_clusters.parquet` | Base com janelas rotuladas com perfis e coordenadas PCA 2D (168 KB) | ✅ Concluído |
| `data/processed/driving_behavior_clusters.csv` | Tabela rotulada para abertura tabular (311 KB) | ✅ Concluído |
| `outputs/models/kmeans_driver_profile.joblib` | Modelo de Machine Learning K-Means treinado e calibrado (3 clusters) | ✅ Concluído |
| `outputs/models/scaler_driver_profile.joblib` | Normalizador StandardScaler ajustado sobre as 6 variáveis comportamentais | ✅ Concluído |
| `outputs/models/pca_driver_profile.joblib` | Modelo de Redução de Dimensionalidade PCA (2 componentes, 69.4% variância) | ✅ Concluído |
| `outputs/results/clustering_metrics.json` | Métricas numéricas de avaliação (Inércia, Silhueta, Davies-Bouldin, Calinski) | ✅ Concluído |
| `outputs/figures/elbow_and_silhouette_analysis.png` | Gráfico acadêmico da Curva do Cotovelo e Coeficiente de Silhueta (300 DPI) | ✅ Concluído |
| `outputs/figures/driving_clusters_pca_2d.png` | Visualização dos clusters no espaço bidimensional do PCA (300 DPI) | ✅ Concluído |
| `outputs/figures/cluster_profiles_comparison.png` | Gráfico de barras comparando as métricas cinemáticas médias por perfil (300 DPI) | ✅ Concluído |
| `requirements.txt` | Lista de bibliotecas e dependências do ambiente Python | ✅ Concluído |
| `VED-master/` | Repositório original com os 54 arquivos de dados semanais do VED | ✅ Disponível |

---

## 📝 Histórico de Execuções e Marcos de Desenvolvimento

### [Marco 0: Análise, Planejamento e Diagnóstico de Ambiente] - 02/09/2026
* **Ações Realizadas:**
  * Leitura e extração integral do documento de Pré-banca (7 páginas).
  * Inspeção do dataset VED: identificação de 54 arquivos dinâmicos semanais e arquivos estáticos de metadados.
  * Alinhamento metodológico com o autor (foco exclusivo em EVs puros, prioridade na classificação de perfis com teste interativo).
  * Criação dos documentos oficiais de planejamento `docs/PLANO_DE_IMPLEMENTACAO.md`, `README.md` e `docs/REGISTRO_DE_DESENVOLVIMENTO.md`.
  * Diagnóstico de Ambiente Concluído com Sucesso e criação de `requirements.txt`.

### [Marco 1: Execução da Etapa 1.1 - Extração dos Dados EV] - 02/09/2026
* **Ações Realizadas:**
  * Estruturação física de diretórios e módulo `src/config.py`.
  * Extração consolidada dos veículos 100% elétricos (IDs 10, 455 e 541) via `src/data_loader.py`.
  * Validação com **476.308 registros**, **491 viagens**, 0 nulos e criação do validador `test_step1.py`.

### [Marco 2: Execução da Etapa 1.2 - Engenharia de Atributos Comportamentais] - 19/09/2026 e 20/09/2026
* **Ações Realizadas:**
  * Implementação do módulo `src/feature_engineering.py`.
  * Regularização temporal a 1 Hz exato e correção de ruído de clock CAN do OBD-II.
  * Agrupamento estrito por chave composta `(VehId, Trip)` evitando contaminação entre carros.
  * Segmentação em janelas operacionais de 120 segundos (Mobini Seraji et al., 2025).
  * Extração de métricas cinemáticas normalizadas por minuto (freadas bruscas, arrancadas rápidas, oscilação de velocidade, potência em kW).
  * Validação completa via `test_step1_2.py` e criação de `inspect_trip.py`.

### [Marco 3: Execução da Etapa 1.3 - Clusterização K-Means e Classificação de Perfis] - 20/09/2026
* **Ações Realizadas:**
  * Implementação do pipeline de clusterização em `src/clustering.py`.
  * Filtragem de trechos com velocidade média $\ge 8\text{ km/h}$ para avaliar estritamente condução ativa (2.481 janelas operacionais).
  * Padronização via `StandardScaler` sobre 6 variáveis comportamentais essenciais.
  * Varredura paramétrica de $k \in [2, 6]$ avaliada com Inércia (Método do Cotovelo), Coeficiente de Silhueta, Davies-Bouldin Index e Calinski-Harabasz Index.
  * Seleção de $k=3$ (Econômico / Suave, Moderado / Regular, Agressivo / Dinâmico).
  * Implementação de classe invólucro determinística `DrivingProfileModel` para assegurar ordenação semântica e serialização segura sem violar atributos privados do Scikit-Learn.
  * Redução de dimensionalidade via PCA (2 componentes capturando 69.4% da variância total) para inspeção visual 2D.
  * Exportação de 3 figuras acadêmicas em alta resolução (300 DPI) para artigo/monografia:
    * `elbow_and_silhouette_analysis.png`
    * `driving_clusters_pca_2d.png`
    * `cluster_profiles_comparison.png`
  * Criação do script de testes automatizados `test_step1_3.py` com validação de arquivos, métricas, distribuição e 3 cenários de teste de generalização com 100% de sucesso.
* **Distribuição Final dos Perfis:**
  * **Moderado / Regular:** 1.377 trechos (55,5%)
  * **Econômico / Suave:** 690 trechos (27,8%)
  * **Agressivo / Dinâmico:** 414 trechos (16,7%)
* **Próxima Ação:** Desenvolvimento da aplicação visual interativa (Streamlit / Dashboard) para exploração das viagens, dos gráficos e classificação em tempo real de novos perfis.

