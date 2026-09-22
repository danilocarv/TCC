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

# Caminhos dos artefatos da Etapa 1.2 (Engenharia de Atributos Comportamentais)
PROCESSED_BEHAVIOR_PARQUET = PROCESSED_DATA_DIR / "driving_behavior_features.parquet"
PROCESSED_BEHAVIOR_CSV = PROCESSED_DATA_DIR / "driving_behavior_features.csv"
PROCESSED_BEHAVIOR_METADATA_JSON = PROCESSED_DATA_DIR / "driving_behavior_metadata.json"

# Caminhos dos artefatos da Etapa 1.3 (Clusterização K-Means)
PROCESSED_CLUSTERS_PARQUET = PROCESSED_DATA_DIR / "driving_behavior_clusters.parquet"
PROCESSED_CLUSTERS_CSV = PROCESSED_DATA_DIR / "driving_behavior_clusters.csv"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_driver_profile.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "scaler_driver_profile.joblib"
PCA_MODEL_PATH = MODELS_DIR / "pca_driver_profile.joblib"
CLUSTERING_METRICS_JSON = RESULTS_DIR / "clustering_metrics.json"

# Figuras geradas para a monografia / artigo
FIG_ELBOW_SILHOUETTE = FIGURES_DIR / "elbow_and_silhouette_analysis.png"
FIG_CLUSTERS_PCA_2D = FIGURES_DIR / "driving_clusters_pca_2d.png"
FIG_CLUSTER_PROFILES = FIGURES_DIR / "cluster_profiles_comparison.png"

# Atributos comportamentais selecionados para a clusterização
CLUSTER_FEATURE_COLS = [
    "mean_speed_kmh",
    "std_speed_kmh",
    "mean_pos_accel_ms2",
    "max_pos_accel_ms2",
    "hard_braking_rate_min",
    "rapid_accel_rate_min",
]

# Parâmetros de janelamento e cinemática
WINDOW_SIZE_SECONDS = 120  # Janelas de 2 minutos (120 segundos) conforme literatura
MIN_WINDOW_POINTS = 30     # Mínimo de 30 segundos de dados válidos para formar uma janela
HARD_BRAKING_THRESHOLD = -2.0  # Limiar de frenagem brusca em m/s²
RAPID_ACCEL_THRESHOLD = 2.0    # Limiar de aceleração brusca em m/s²

# Garantir criação automática das pastas necessárias
def ensure_directories():
    """Garante que todas as pastas essenciais do projeto existam."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
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

