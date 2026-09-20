# Diário de Bordo e Mapeamento do Projeto (TCC)

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Projeto:** Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos  
**Status Atual:** Fase 1 - Etapa 1.2 concluída com sucesso (Aguardando validação para Etapa 1.3)  

---

## 🗺️ Mapa de Arquivos e Componentes do Projeto

Abaixo encontra-se o inventário atualizado de todos os componentes do repositório:

| Caminho / Arquivo | Descrição | Status |
| :--- | :--- | :--- |
| `docs/Pré-banca de TCC - Danilo Carvalho de Oliveira.pdf` | Documento original da proposta de TCC apresentada à banca | ✅ Concluído |
| `docs/PLANO_DE_IMPLEMENTACAO.md` | Plano mestre detalhado contendo a arquitetura e divisão das fases | ✅ Concluído |
| `docs/REGISTRO_DE_DESENVOLVIMENTO.md` | Este diário de bordo com o histórico e mapeamento do projeto | ✅ Ativo |
| `docs/HISTORICO_TECNICO_E_AJUSTES.md` | Relatório técnico detalhado com histórico de decisões, causas-raízes e correções | ✅ Ativo |
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
| `data/processed/driving_behavior_features.parquet` | Base com os atributos comportamentais das 2.531 janelas de condução (135 KB) | ✅ Concluído |
| `data/processed/driving_behavior_features.csv` | Tabela dos atributos comportamentais para abertura no Excel/VS Code (240 KB) | ✅ Concluído |
| `data/processed/driving_behavior_metadata.json` | Metadados descritivos das variáveis comportamentais | ✅ Concluído |
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

### [Marco 2: Execução da Etapa 1.2 - Engenharia de Atributos Comportamentais] - 19/09/2026
* **Ações Realizadas:**
  * Implementação do módulo `src/feature_engineering.py`.
  * Cálculo de variáveis cinemáticas segundo a segundo: aceleração longitudinal ($a = \Delta v / \Delta t$) com suavização móvel de 3 pontos para remoção de ruído de degrau dos sensores OBD-II, e cálculo de *jerk* ($j = \Delta a / \Delta t$).
  * Segmentação temporal das viagens em janelas operacionais de 120 segundos (conforme fundamentação teórica de Mobini Seraji et al., 2025).
  * Extração de métricas-chave do condutor: velocidade média/máxima, desvio padrão da velocidade (estabilidade), aceleração positiva média, pico de aceleração/frenagem, taxas de frenagens/acelerações bruscas por minuto ($|a| > 2.0\text{ m/s}^2$), eventos de alto jerk e taxa de tempo ocioso (*idle ratio*).
  * Criação e validação do script `test_step1_2.py`.
* **Resultados Obtidos:**
  * **2.531 trechos operacionais de condução** gerados e prontos para clusterização.
  * **491 viagens reais** representadas.
  * **0 valores nulos**.
  * Arquivos gerados em `data/processed/`: `driving_behavior_features.parquet` (135 KB), `driving_behavior_features.csv` (240 KB) e `driving_behavior_metadata.json`.
* **Próxima Ação:** Danilo testar a Etapa 1.2 via `python test_step1_2.py` e autorizar a Etapa 1.3 (Clusterização Não Supervisionada com K-Means: Curva do Cotovelo, Coeficiente de Silhueta e Rotulagem dos Perfis).
