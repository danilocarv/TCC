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

# Caminhos dos artefatos de DTW (1700x1700 sobre velocidade no treino)
DTW_SPEED_MATRIX_NPY = PROCESSED_DATA_DIR / "dtw_speed_matrix_train_1700x1700.npy"
DTW_SPEED_MATRIX_PARQUET = PROCESSED_DATA_DIR / "dtw_speed_matrix_train_1700x1700.parquet"
FIG_DTW_SPEED_HEATMAP = FIGURES_DIR / "07_dtw_speed_heatmap_1700x1700.png"

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


if __name__ == "__main__":
    ensure_directories()
    print("Pastas do projeto estruturadas com sucesso!")

