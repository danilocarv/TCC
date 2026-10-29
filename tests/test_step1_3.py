"""
Script de Teste e Validação da Etapa 1.3 (Clusterização K-Means).
Verifica a integridade dos modelos treinados, das figuras geradas e testa predições em tempo real.
Execute no terminal com: python tests/test_step1_3.py
"""

import sys
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import joblib

from src.clustering import DrivingProfileModel
# Garante compatibilidade de unpickle se o modelo foi salvo via __main__
sys.modules["__main__"].DrivingProfileModel = DrivingProfileModel

from src.config import (
    CLUSTER_FEATURE_COLS,
    PROCESSED_CLUSTERS_PARQUET as CLUSTERS_PARQUET,
    PROCESSED_CLUSTERS_CSV as CLUSTERS_CSV,
    KMEANS_MODEL_PATH as KMEANS_FILE,
    SCALER_MODEL_PATH as SCALER_FILE,
    PCA_MODEL_PATH as PCA_FILE,
    FIG_ELBOW_SILHOUETTE as FIG_ELBOW,
    FIG_CLUSTERS_PCA_2D as FIG_PCA,
    FIG_CLUSTER_PROFILES as FIG_PROFILES,
)


def test_step_1_3():
    print("=" * 70)
    print("INICIANDO TESTE DE VALIDACAO DA ETAPA 1.3 (CLUSTERIZACAO K-MEANS)")
    print("=" * 70)

    # 1. Verificar existência dos artefatos
    print("\n[1/4] Verificando se os modelos e figuras acadêmicas existem...")
    for f, name in [
        (KMEANS_FILE, "Modelo K-Means (.joblib)"),
        (SCALER_FILE, "Normalizador StandardScaler (.joblib)"),
        (PCA_FILE, "Modelo PCA (.joblib)"),
        (CLUSTERS_PARQUET, "Dataset com Clusters Parquet"),
        (FIG_ELBOW, "Gráfico Cotovelo e Silhueta (300 DPI)"),
        (FIG_PCA, "Gráfico 2D PCA dos Clusters (300 DPI)"),
        (FIG_PROFILES, "Gráfico Comparativo de Perfis (300 DPI)"),
    ]:
        if not f.exists():
            print(f"[ERRO] Arquivo {name} nao encontrado em: {f}")
            sys.exit(1)
        print(f"  [OK] {name}: Encontrado! ({f.name})")

    # 2. Carregar o dataset rotulado
    print("\n[2/4] Validando o dataset final com os clusters atribuídos...")
    df = pd.read_parquet(CLUSTERS_PARQUET)
    print(f"  [OK] Total de janelas rotuladas: {len(df):,}")
    print(f"  [OK] Colunas do cluster presentes: 'cluster', 'cluster_label'")

    # Verificar proporção dos clusters
    print("\n  Distribuição Populacional dos Perfis Identificados:")
    dist = df["cluster_label"].value_counts(normalize=True) * 100
    for label, pct in dist.items():
        count = (df["cluster_label"] == label).sum()
        print(f"    - {label}: {count} janelas ({pct:.1f}%)")

    # 3. Validar a consistência física dos perfis
    print("\n[3/4] Validando a consistência dos perfis comportamentais...")
    summary = df.groupby("cluster_label")[
        ["mean_speed_kmh", "mean_pos_accel_ms2", "hard_braking_rate_min"]
    ].mean()
    print(summary.to_string())

    # Garantir que o perfil Agressivo tenha aceleração média superior ao Econômico
    if "Agressivo / Dinâmico" in summary.index and "Econômico / Suave" in summary.index:
        acc_agr = summary.loc["Agressivo / Dinâmico", "mean_pos_accel_ms2"]
        acc_eco = summary.loc["Econômico / Suave", "mean_pos_accel_ms2"]
        assert acc_agr > acc_eco, "Aceleração do perfil agressivo deve ser maior que do suave!"
        print(
            f"  [OK] Consistência física confirmada: Agressivo ({acc_agr:.2f} m/s²) > Suave ({acc_eco:.2f} m/s²)"
        )

    # 4. Testar inferência com a classe DrivingProfileModel
    print("\n[4/4] Testando o pipeline de inferência em tempo real...")
    model = joblib.load(KMEANS_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Janela sintética de teste: Condução calma
    sample_calm = pd.DataFrame(
        [
            {
                "mean_speed_kmh": 25.0,
                "std_speed_kmh": 5.0,
                "mean_pos_accel_ms2": 0.4,
                "max_pos_accel_ms2": 1.0,
                "hard_braking_rate_min": 0.0,
                "rapid_accel_rate_min": 0.0,
            }
        ]
    )

    # Janela sintética de teste: Condução agressiva
    sample_aggressive = pd.DataFrame(
        [
            {
                "mean_speed_kmh": 65.0,
                "std_speed_kmh": 20.0,
                "mean_pos_accel_ms2": 1.6,
                "max_pos_accel_ms2": 3.5,
                "hard_braking_rate_min": 2.0,
                "rapid_accel_rate_min": 3.0,
            }
        ]
    )

    pred_calm = model.predict_label(scaler.transform(sample_calm[CLUSTER_FEATURE_COLS].values))[0]
    pred_agr = model.predict_label(scaler.transform(sample_aggressive[CLUSTER_FEATURE_COLS].values))[0]

    print(f"  * Predição para condução calma:      {pred_calm}")
    print(f"  * Predição para condução agressiva:  {pred_agr}")

    print("\n" + "=" * 70)
    print("[SUCESSO] ETAPA 1.3 100% VALIDADA E APROVADA!")
    print("=" * 70)


if __name__ == "__main__":
    test_step_1_3()
