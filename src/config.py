"""
Módulo de Configuração Global do Projeto de TCC.
Centraliza caminhos de diretórios, identificadores dos veículos elétricos e parâmetros.
"""

from pathlib import Path

# Diretórios principais do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
APP_DIR = BASE_DIR / "app"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Diretórios do dataset original VED
VED_DIR = BASE_DIR / "VED-master"
VED_DATA_DIR = VED_DIR / "Data"
VED_STATIC_PHEV_EV_FILE = VED_DATA_DIR / "VED_Static_Data_PHEV&EV.xlsx"

# Identificadores dos Veículos 100% Elétricos (Nissan Leaf 2013 - 24 kWh)
EV_VEHICLE_IDS = [10, 455, 541]

# Colunas relevantes a extrair dos dados dinâmicos do VED
COLUMNS_DYNAMIC = [
    "DayNum",
    "VehId",
    "Trip",
    "Timestamp(ms)",
    "Latitude[deg]",
    "Longitude[deg]",
    "Vehicle Speed[km/h]",
    "OAT[DegC]",
    "Air Conditioning Power[Watts]",
    "Heater Power[Watts]",
    "HV Battery Current[A]",
    "HV Battery SOC[%]",
    "HV Battery Voltage[V]",
]

# Tipos de dados otimizados para economia de memória
DTYPES_DYNAMIC = {
    "VehId": "int16",
    "Trip": "int32",
    "Timestamp(ms)": "int64",
    "Latitude[deg]": "float32",
    "Longitude[deg]": "float32",
    "Vehicle Speed[km/h]": "float32",
    "OAT[DegC]": "float32",
    "Air Conditioning Power[Watts]": "float32",
    "Heater Power[Watts]": "float32",
    "HV Battery Current[A]": "float32",
    "HV Battery SOC[%]": "float32",
    "HV Battery Voltage[V]": "float32",
}

# Caminhos dos artefatos da Etapa 1.1 (Dados brutos consolidados)
RAW_EV_PARQUET = RAW_DATA_DIR / "ev_telemetry.parquet"
RAW_EV_CSV = RAW_DATA_DIR / "ev_telemetry.csv"
RAW_EV_METADATA_JSON = RAW_DATA_DIR / "ev_dataset_metadata.json"

# Caminhos dos artefatos da Etapa 1.2 (Telemetria 1 Hz, Janelas Individuais e Atributos Comportamentais)
PROCESSED_1HZ_PARQUET = PROCESSED_DATA_DIR / "ev_trips_1hz.parquet"
PROCESSED_BEHAVIOR_PARQUET = PROCESSED_DATA_DIR / "driving_behavior_features.parquet"
PROCESSED_BEHAVIOR_CSV = PROCESSED_DATA_DIR / "driving_behavior_features.csv"
PROCESSED_BEHAVIOR_METADATA_JSON = PROCESSED_DATA_DIR / "driving_behavior_metadata.json"

# Diretórios para os 2.234 arquivos individuais de janelas (1.700 Treino / 534 Teste em .csv e .parquet)
WINDOWS_DIR = DATA_DIR / "windows"
WINDOWS_TRAIN_DIR = WINDOWS_DIR / "train"
WINDOWS_TRAIN_CSV_DIR = WINDOWS_TRAIN_DIR / "csv"
WINDOWS_TRAIN_PARQUET_DIR = WINDOWS_TRAIN_DIR / "parquet"
WINDOWS_TEST_DIR = WINDOWS_DIR / "test"
WINDOWS_TEST_CSV_DIR = WINDOWS_TEST_DIR / "csv"
WINDOWS_TEST_PARQUET_DIR = WINDOWS_TEST_DIR / "parquet"

# Manifesto e tabelas de features separadas em Treino (1.700) e Teste (534)
WINDOW_SPLIT_MANIFEST_CSV = PROCESSED_DATA_DIR / "window_split_manifest.csv"
PROCESSED_TRAIN_FEATURES_PARQUET = PROCESSED_DATA_DIR / "driving_windows_train_features.parquet"
PROCESSED_TRAIN_FEATURES_CSV = PROCESSED_DATA_DIR / "driving_windows_train_features.csv"
PROCESSED_TEST_FEATURES_PARQUET = PROCESSED_DATA_DIR / "driving_windows_test_features.parquet"
PROCESSED_TEST_FEATURES_CSV = PROCESSED_DATA_DIR / "driving_windows_test_features.csv"

# Caminhos dos artefatos de DTW (Piloto 100x100 e Matriz Completa 1700x1700 sobre velocidade)
DTW_PILOT_SIZE = 100
DTW_PILOT_MATRIX_NPY = PROCESSED_DATA_DIR / "dtw_speed_matrix_pilot_100x100.npy"
DTW_PILOT_MATRIX_PARQUET = PROCESSED_DATA_DIR / "dtw_speed_matrix_pilot_100x100.parquet"
FIG_DTW_PILOT_HEATMAP = FIGURES_DIR / "07_dtw_speed_heatmap_pilot_100x100.png"

DTW_SPEED_MATRIX_NPY = PROCESSED_DATA_DIR / "dtw_speed_matrix_train_1700x1700.npy"
DTW_SPEED_MATRIX_PARQUET = PROCESSED_DATA_DIR / "dtw_speed_matrix_train_1700x1700.parquet"
FIG_DTW_SPEED_HEATMAP = FIGURES_DIR / "07_dtw_speed_heatmap_1700x1700.png"

# Caminhos dos artefatos de Detecção de Comunidades (Algoritmo de Louvain sobre Grafo DTW)
LOUVAIN_PILOT_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_louvain_clusters_pilot_100x100.csv"
LOUVAIN_PILOT_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_louvain_clusters_pilot_100x100.parquet"
FIG_LOUVAIN_PILOT_HEATMAP = FIGURES_DIR / "08_dtw_louvain_heatmap_pilot_100x100.png"

LOUVAIN_FULL_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_louvain_clusters_train_1700x1700.csv"
LOUVAIN_FULL_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_louvain_clusters_train_1700x1700.parquet"
FIG_LOUVAIN_FULL_HEATMAP = FIGURES_DIR / "08_dtw_louvain_heatmap_train_1700x1700.png"
LOUVAIN_KNN_NEIGHBORS = 15      # Número de vizinhos mais próximos no grafo de afinidade DTW (k-NN)
LOUVAIN_RANDOM_SEED = 42        # Semente pseudoaleatória para reprodutibilidade do Louvain

# Caminhos dos artefatos da Árvore Geradora Mínima (MST Pura sobre Matriz DTW - Etapa 1)
MST_PILOT_EDGES_CSV = PROCESSED_DATA_DIR / "dtw_mst_edges_pilot_100x100.csv"
MST_PILOT_EDGES_PARQUET = PROCESSED_DATA_DIR / "dtw_mst_edges_pilot_100x100.parquet"
MST_PILOT_MATRIX_NPY = PROCESSED_DATA_DIR / "dtw_mst_matrix_pilot_100x100.npy"
FIG_MST_PURE_PILOT = FIGURES_DIR / "09_dtw_mst_pure_tree_pilot_100x100.png"

MST_FULL_EDGES_CSV = PROCESSED_DATA_DIR / "dtw_mst_edges_train_1700x1700.csv"
MST_FULL_EDGES_PARQUET = PROCESSED_DATA_DIR / "dtw_mst_edges_train_1700x1700.parquet"
MST_FULL_MATRIX_NPY = PROCESSED_DATA_DIR / "dtw_mst_matrix_train_1700x1700.npy"
FIG_MST_PURE_FULL = FIGURES_DIR / "09_dtw_mst_pure_tree_train_1700x1700.png"
MST_STRUCTURAL_METRICS_JSON = RESULTS_DIR / "mst_structural_metrics.json"

# Caminhos dos artefatos do Algoritmo de Louvain Puro sobre a MST (Etapa 2)
LOUVAIN_MST_PILOT_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_mst_louvain_clusters_pilot_100x100.csv"
LOUVAIN_MST_PILOT_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_mst_louvain_clusters_pilot_100x100.parquet"
FIG_LOUVAIN_MST_PILOT_HEATMAP = FIGURES_DIR / "10_dtw_mst_louvain_heatmap_pilot_100x100.png"
FIG_LOUVAIN_MST_TOPOLOGY_PILOT = FIGURES_DIR / "11_dtw_mst_louvain_topology_pilot_100x100.png"

LOUVAIN_MST_FULL_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_mst_louvain_clusters_train_1700x1700.csv"
LOUVAIN_MST_FULL_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_mst_louvain_clusters_train_1700x1700.parquet"
FIG_LOUVAIN_MST_FULL_HEATMAP = FIGURES_DIR / "10_dtw_mst_louvain_heatmap_train_1700x1700.png"
FIG_LOUVAIN_MST_TOPOLOGY_FULL = FIGURES_DIR / "11_dtw_mst_louvain_topology_train_1700x1700.png"
LOUVAIN_MST_COMPARISON_JSON = RESULTS_DIR / "mst_vs_full_louvain_comparison.json"
LOUVAIN_MST_COMPARISON_CSV = RESULTS_DIR / "mst_vs_full_louvain_comparison.csv"

# Caminhos dos artefatos da Filtragem Direta por Percentil 25 de Distância DTW + Louvain Puro (Etapa 3A)
DTW_PERCENTILE_CUTOFF = 25.0
P25_DIRECT_PILOT_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_p25_direct_clusters_pilot_100x100.csv"
P25_DIRECT_PILOT_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_p25_direct_clusters_pilot_100x100.parquet"
FIG_P25_DIRECT_PILOT_HEATMAP = FIGURES_DIR / "12_dtw_p25_direct_heatmap_pilot_100x100.png"

P25_DIRECT_FULL_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_p25_direct_clusters_train_1700x1700.csv"
P25_DIRECT_FULL_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_p25_direct_clusters_train_1700x1700.parquet"
FIG_P25_DIRECT_FULL_HEATMAP = FIGURES_DIR / "12_dtw_p25_direct_heatmap_train_1700x1700.png"
P25_DIRECT_METRICS_JSON = RESULTS_DIR / "dtw_p25_direct_metrics.json"
P25_DIRECT_SUMMARY_CSV = RESULTS_DIR / "dtw_p25_direct_summary.csv"

# Caminhos dos artefatos da Rede Otimizada: União da MST com o Percentil 25 (MST ∪ P25) + Louvain Puro (Etapa 3B)
P25_MST_PILOT_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_p25_mst_clusters_pilot_100x100.csv"
P25_MST_PILOT_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_p25_mst_clusters_pilot_100x100.parquet"
FIG_P25_MST_PILOT_HEATMAP = FIGURES_DIR / "13_dtw_p25_mst_heatmap_pilot_100x100.png"

P25_MST_FULL_CLUSTERS_CSV = PROCESSED_DATA_DIR / "dtw_p25_mst_clusters_train_1700x1700.csv"
P25_MST_FULL_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "dtw_p25_mst_clusters_train_1700x1700.parquet"
FIG_P25_MST_FULL_HEATMAP = FIGURES_DIR / "13_dtw_p25_mst_heatmap_train_1700x1700.png"
P25_MST_METRICS_JSON = RESULTS_DIR / "dtw_p25_mst_metrics.json"
P25_MST_SUMMARY_CSV = RESULTS_DIR / "dtw_p25_mst_summary.csv"

# Relatório e Tabela Comparativa Geral de Todos os Cenários (Grafo Completo, MST Pura, P25 Direto, MST ∪ P25)
DTW_ALL_STAGES_COMPARISON_JSON = RESULTS_DIR / "dtw_all_stages_comparison.json"
DTW_ALL_STAGES_COMPARISON_CSV = RESULTS_DIR / "dtw_all_stages_comparison.csv"

# Comparativo Mestre Final: K-Means vs. Redes Complexas (Grafo Completo, MST, P25, MST ∪ P25)
MASTER_COMPARISON_CSV = RESULTS_DIR / "master_methodology_comparison.csv"
MASTER_COMPARISON_JSON = RESULTS_DIR / "master_methodology_comparison.json"
KMEANS_LOUVAIN_CONTINGENCY_CSV = RESULTS_DIR / "contingency_kmeans_vs_louvain.csv"
FIG_FINAL_COMPARISON_SYNTHESIS = FIGURES_DIR / "14_methodology_comparison_synthesis.png"


# Caminhos dos artefatos da Etapa 1.3 (Clusterização K-Means)
PROCESSED_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "driving_behavior_clusters.parquet"
PROCESSED_CLUSTERS_CSV = PROCESSED_DATA_DIR / "driving_behavior_clusters.csv"
PROCESSED_TRAIN_LABELED_PARQUET = PROCESSED_DATA_DIR / "driving_windows_train_labeled.parquet"
PROCESSED_TRAIN_LABELED_CSV = PROCESSED_DATA_DIR / "driving_windows_train_labeled.csv"
PROCESSED_TEST_LABELED_PARQUET = PROCESSED_DATA_DIR / "driving_windows_test_labeled.parquet"
PROCESSED_TEST_LABELED_CSV = PROCESSED_DATA_DIR / "driving_windows_test_labeled.csv"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_driver_profile.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "scaler_driver_profile.joblib"
PCA_MODEL_PATH = MODELS_DIR / "pca_driver_profile.joblib"
CLUSTERING_METRICS_JSON = RESULTS_DIR / "clustering_metrics.json"

# Modelos e validação out-of-sample do K-Means no split Treino (1.700) e Teste (534)
KMEANS_TRAIN_MODEL_PATH = MODELS_DIR / "kmeans_train_1700.joblib"
SCALER_TRAIN_MODEL_PATH = MODELS_DIR / "scaler_train_1700.joblib"
PCA_TRAIN_MODEL_PATH = MODELS_DIR / "pca_train_1700.joblib"
KMEANS_TRAIN_TEST_VALIDATION_JSON = RESULTS_DIR / "kmeans_train_test_validation.json"
KMEANS_TRAIN_TEST_VALIDATION_CSV = RESULTS_DIR / "kmeans_train_test_validation.csv"
FIG_KMEANS_TRAIN_TEST_PCA = FIGURES_DIR / "15_kmeans_train_test_pca_comparison.png"
FIG_KMEANS_TRAIN_TEST_PROFILES = FIGURES_DIR / "16_kmeans_train_test_profiles_comparison.png"

# Figuras geradas para a monografia / artigo
FIG_ELBOW_SILHOUETTE = FIGURES_DIR / "elbow_and_silhouette_analysis.png"
FIG_CLUSTERS_PCA_2D = FIGURES_DIR / "driving_clusters_pca_2d.png"
FIG_CLUSTER_PROFILES = FIGURES_DIR / "cluster_profiles_comparison.png"

# Atributos comportamentais selecionados para a clusterização (Opção A: 6 features cinemáticas principais)
CLUSTER_FEATURE_COLS = [
    "mean_speed_kmh",
    "std_speed_kmh",
    "mean_pos_accel_ms2",
    "max_pos_accel_ms2",
    "hard_braking_rate_min",
    "rapid_accel_rate_min",
]

# Atributos derivados adicionais por janela (espaciais + dinâmicos complementares)
DERIVED_CONTEXT_COLS = [
    "distance_total_m",
    "distance_straight_m",
    "tortuosity_index",
    "mean_neg_accel_ms2",
    "min_neg_accel_ms2",
    "std_jerk_ms3",
    "high_jerk_rate_min",
    "idle_ratio",
]

# Parâmetros de janelamento, condução ativa e divisão Treino/Teste
WINDOW_SIZE_SECONDS = 120  # Janelas de 2 minutos (120 segundos) conforme literatura
MIN_WINDOW_POINTS = 30     # Mínimo de 30 segundos de dados válidos para formar uma janela
ACTIVE_SPEED_THRESHOLD_KMH = 8.0  # Velocidade média mínima para caracterizar condução ativa
TRAIN_WINDOW_COUNT = 1700  # Quantidade exata de janelas ativas para treino
TEST_WINDOW_COUNT = 534    # Quantidade exata de janelas ativas para teste (1700 + 534 = 2234)
RANDOM_SEED = 42           # Semente fixa para embaralhamento reprodutível (shuffle)
HARD_BRAKING_THRESHOLD = -2.0  # Limiar de frenagem brusca em m/s²
RAPID_ACCEL_THRESHOLD = 2.0    # Limiar de aceleração brusca em m/s²
HIGH_JERK_THRESHOLD = 3.0      # Limiar de tranco longitudinal / jerk severo em m/s³

# Garantir criação automática das pastas necessárias
def ensure_directories():
    """Garante que todas as pastas essenciais do projeto existam."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        WINDOWS_TRAIN_CSV_DIR,
        WINDOWS_TRAIN_PARQUET_DIR,
        WINDOWS_TEST_CSV_DIR,
        WINDOWS_TEST_PARQUET_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        APP_DIR,
        NOTEBOOKS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# Registro centralizado de metadados de séries temporais suportadas pelo pipeline de grafos/DTW
FEATURE_REGISTRY = {
    "speed_kmh": {
        "key": "speed_kmh",
        "column": "Vehicle Speed[km/h]",
        "name_pt": "Velocidade",
        "unit": "km/h",
    },
    "accel_ms2": {
        "key": "accel_ms2",
        "column": "accel_ms2",
        "name_pt": "Aceleração Longitudinal",
        "unit": "m/s²",
    },
    "power_kw": {
        "key": "power_kw",
        "column": "power_kw",
        "name_pt": "Potência Elétrica",
        "unit": "kW",
    },
}


def get_feature_paths(feature: str = "speed_kmh") -> dict:
    """
    Retorna os metadados e todos os caminhos de artefatos (matrizes, grafos, figuras e resultados)
    para a feature informada. Para 'speed_kmh', preserva estritamente todos os caminhos originais.
    """
    if feature not in FEATURE_REGISTRY:
        raise ValueError(
            f"Feature '{feature}' não reconhecida. Opções válidas: {list(FEATURE_REGISTRY.keys())}"
        )

    meta = FEATURE_REGISTRY[feature]
    if feature == "speed_kmh":
        return {
            **meta,
            "dtw_pilot_npy": DTW_PILOT_MATRIX_NPY,
            "dtw_pilot_parquet": DTW_PILOT_MATRIX_PARQUET,
            "fig_dtw_pilot_heatmap": FIG_DTW_PILOT_HEATMAP,
            "dtw_full_npy": DTW_SPEED_MATRIX_NPY,
            "dtw_full_parquet": DTW_SPEED_MATRIX_PARQUET,
            "fig_dtw_full_heatmap": FIG_DTW_SPEED_HEATMAP,
            "louvain_pilot_csv": LOUVAIN_PILOT_CLUSTERS_CSV,
            "louvain_pilot_parquet": LOUVAIN_PILOT_CLUSTERS_PARQUET,
            "fig_louvain_pilot_heatmap": FIG_LOUVAIN_PILOT_HEATMAP,
            "louvain_full_csv": LOUVAIN_FULL_CLUSTERS_CSV,
            "louvain_full_parquet": LOUVAIN_FULL_CLUSTERS_PARQUET,
            "fig_louvain_full_heatmap": FIG_LOUVAIN_FULL_HEATMAP,
            "mst_pilot_edges_csv": MST_PILOT_EDGES_CSV,
            "mst_pilot_edges_parquet": MST_PILOT_EDGES_PARQUET,
            "mst_pilot_matrix_npy": MST_PILOT_MATRIX_NPY,
            "fig_mst_pure_pilot": FIG_MST_PURE_PILOT,
            "mst_full_edges_csv": MST_FULL_EDGES_CSV,
            "mst_full_edges_parquet": MST_FULL_EDGES_PARQUET,
            "mst_full_matrix_npy": MST_FULL_MATRIX_NPY,
            "fig_mst_pure_full": FIG_MST_PURE_FULL,
            "mst_structural_metrics_json": MST_STRUCTURAL_METRICS_JSON,
            "louvain_mst_pilot_csv": LOUVAIN_MST_PILOT_CLUSTERS_CSV,
            "louvain_mst_pilot_parquet": LOUVAIN_MST_PILOT_CLUSTERS_PARQUET,
            "fig_louvain_mst_pilot_heatmap": FIG_LOUVAIN_MST_PILOT_HEATMAP,
            "fig_louvain_mst_topology_pilot": FIG_LOUVAIN_MST_TOPOLOGY_PILOT,
            "louvain_mst_full_csv": LOUVAIN_MST_FULL_CLUSTERS_CSV,
            "louvain_mst_full_parquet": LOUVAIN_MST_FULL_CLUSTERS_PARQUET,
            "fig_louvain_mst_full_heatmap": FIG_LOUVAIN_MST_FULL_HEATMAP,
            "fig_louvain_mst_topology_full": FIG_LOUVAIN_MST_TOPOLOGY_FULL,
            "louvain_mst_comparison_json": LOUVAIN_MST_COMPARISON_JSON,
            "louvain_mst_comparison_csv": LOUVAIN_MST_COMPARISON_CSV,
            "p25_direct_pilot_csv": P25_DIRECT_PILOT_CLUSTERS_CSV,
            "p25_direct_pilot_parquet": P25_DIRECT_PILOT_CLUSTERS_PARQUET,
            "fig_p25_direct_pilot_heatmap": FIG_P25_DIRECT_PILOT_HEATMAP,
            "p25_direct_full_csv": P25_DIRECT_FULL_CLUSTERS_CSV,
            "p25_direct_full_parquet": P25_DIRECT_FULL_CLUSTERS_PARQUET,
            "fig_p25_direct_full_heatmap": FIG_P25_DIRECT_FULL_HEATMAP,
            "p25_direct_metrics_json": P25_DIRECT_METRICS_JSON,
            "p25_direct_summary_csv": P25_DIRECT_SUMMARY_CSV,
            "p25_mst_pilot_csv": P25_MST_PILOT_CLUSTERS_CSV,
            "p25_mst_pilot_parquet": P25_MST_PILOT_CLUSTERS_PARQUET,
            "fig_p25_mst_pilot_heatmap": FIG_P25_MST_PILOT_HEATMAP,
            "p25_mst_full_csv": P25_MST_FULL_CLUSTERS_CSV,
            "p25_mst_full_parquet": P25_MST_FULL_CLUSTERS_PARQUET,
            "fig_p25_mst_full_heatmap": FIG_P25_MST_FULL_HEATMAP,
            "p25_mst_metrics_json": P25_MST_METRICS_JSON,
            "p25_mst_summary_csv": P25_MST_SUMMARY_CSV,
            "dtw_all_stages_comparison_json": DTW_ALL_STAGES_COMPARISON_JSON,
            "dtw_all_stages_comparison_csv": DTW_ALL_STAGES_COMPARISON_CSV,
            "master_comparison_csv": MASTER_COMPARISON_CSV,
            "master_comparison_json": MASTER_COMPARISON_JSON,
            "fig_final_comparison_synthesis": FIG_FINAL_COMPARISON_SYNTHESIS,
        }

    fig_sub = FIGURES_DIR / feature
    fig_sub.mkdir(parents=True, exist_ok=True)

    return {
        **meta,
        "dtw_pilot_npy": PROCESSED_DATA_DIR / f"dtw_{feature}_matrix_pilot_100x100.npy",
        "dtw_pilot_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_matrix_pilot_100x100.parquet",
        "fig_dtw_pilot_heatmap": fig_sub / f"07_dtw_{feature}_heatmap_pilot_100x100.png",
        "dtw_full_npy": PROCESSED_DATA_DIR / f"dtw_{feature}_matrix_train_1700x1700.npy",
        "dtw_full_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_matrix_train_1700x1700.parquet",
        "fig_dtw_full_heatmap": fig_sub / f"07_dtw_{feature}_heatmap_1700x1700.png",
        "louvain_pilot_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_louvain_clusters_pilot_100x100.csv",
        "louvain_pilot_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_louvain_clusters_pilot_100x100.parquet",
        "fig_louvain_pilot_heatmap": fig_sub / f"08_dtw_{feature}_louvain_heatmap_pilot_100x100.png",
        "louvain_full_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_louvain_clusters_train_1700x1700.csv",
        "louvain_full_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_louvain_clusters_train_1700x1700.parquet",
        "fig_louvain_full_heatmap": fig_sub / f"08_dtw_{feature}_louvain_heatmap_train_1700x1700.png",
        "mst_pilot_edges_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_edges_pilot_100x100.csv",
        "mst_pilot_edges_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_edges_pilot_100x100.parquet",
        "mst_pilot_matrix_npy": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_matrix_pilot_100x100.npy",
        "fig_mst_pure_pilot": fig_sub / f"09_dtw_{feature}_mst_pure_tree_pilot_100x100.png",
        "mst_full_edges_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_edges_train_1700x1700.csv",
        "mst_full_edges_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_edges_train_1700x1700.parquet",
        "mst_full_matrix_npy": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_matrix_train_1700x1700.npy",
        "fig_mst_pure_full": fig_sub / f"09_dtw_{feature}_mst_pure_tree_train_1700x1700.png",
        "mst_structural_metrics_json": RESULTS_DIR / f"mst_structural_metrics_{feature}.json",
        "louvain_mst_pilot_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_louvain_clusters_pilot_100x100.csv",
        "louvain_mst_pilot_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_louvain_clusters_pilot_100x100.parquet",
        "fig_louvain_mst_pilot_heatmap": fig_sub / f"10_dtw_{feature}_mst_louvain_heatmap_pilot_100x100.png",
        "fig_louvain_mst_topology_pilot": fig_sub / f"11_dtw_{feature}_mst_louvain_topology_pilot_100x100.png",
        "louvain_mst_full_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_louvain_clusters_train_1700x1700.csv",
        "louvain_mst_full_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_mst_louvain_clusters_train_1700x1700.parquet",
        "fig_louvain_mst_full_heatmap": fig_sub / f"10_dtw_{feature}_mst_louvain_heatmap_train_1700x1700.png",
        "fig_louvain_mst_topology_full": fig_sub / f"11_dtw_{feature}_mst_louvain_topology_train_1700x1700.png",
        "louvain_mst_comparison_json": RESULTS_DIR / f"mst_vs_full_louvain_comparison_{feature}.json",
        "louvain_mst_comparison_csv": RESULTS_DIR / f"mst_vs_full_louvain_comparison_{feature}.csv",
        "p25_direct_pilot_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_direct_clusters_pilot_100x100.csv",
        "p25_direct_pilot_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_direct_clusters_pilot_100x100.parquet",
        "fig_p25_direct_pilot_heatmap": fig_sub / f"12_dtw_{feature}_p25_direct_heatmap_pilot_100x100.png",
        "p25_direct_full_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_direct_clusters_train_1700x1700.csv",
        "p25_direct_full_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_direct_clusters_train_1700x1700.parquet",
        "fig_p25_direct_full_heatmap": fig_sub / f"12_dtw_{feature}_p25_direct_heatmap_train_1700x1700.png",
        "p25_direct_metrics_json": RESULTS_DIR / f"dtw_p25_direct_metrics_{feature}.json",
        "p25_direct_summary_csv": RESULTS_DIR / f"dtw_p25_direct_summary_{feature}.csv",
        "p25_mst_pilot_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_mst_clusters_pilot_100x100.csv",
        "p25_mst_pilot_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_mst_clusters_pilot_100x100.parquet",
        "fig_p25_mst_pilot_heatmap": fig_sub / f"13_dtw_{feature}_p25_mst_heatmap_pilot_100x100.png",
        "p25_mst_full_csv": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_mst_clusters_train_1700x1700.csv",
        "p25_mst_full_parquet": PROCESSED_DATA_DIR / f"dtw_{feature}_p25_mst_clusters_train_1700x1700.parquet",
        "fig_p25_mst_full_heatmap": fig_sub / f"13_dtw_{feature}_p25_mst_heatmap_train_1700x1700.png",
        "p25_mst_metrics_json": RESULTS_DIR / f"dtw_p25_mst_metrics_{feature}.json",
        "p25_mst_summary_csv": RESULTS_DIR / f"dtw_p25_mst_summary_{feature}.csv",
        "dtw_all_stages_comparison_json": RESULTS_DIR / f"dtw_all_stages_comparison_{feature}.json",
        "dtw_all_stages_comparison_csv": RESULTS_DIR / f"dtw_all_stages_comparison_{feature}.csv",
        "master_comparison_csv": RESULTS_DIR / f"master_methodology_comparison_{feature}.csv",
        "master_comparison_json": RESULTS_DIR / f"master_methodology_comparison_{feature}.json",
        "fig_final_comparison_synthesis": fig_sub / f"14_methodology_comparison_synthesis_{feature}.png",
    }


if __name__ == "__main__":
    ensure_directories()
    print("Pastas do projeto estruturadas com sucesso!")

