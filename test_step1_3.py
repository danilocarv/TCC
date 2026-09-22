"""
Script de Teste e Validação da Etapa 1.3 (Clusterização K-Means).
Verifica a integridade dos modelos treinados, das figuras geradas e testa predições em tempo real.
Execute no terminal com: python test_step1_3.py
"""

import sys
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib

from src.clustering import DrivingProfileModel

ROOT_DIR = Path(__file__).resolve().parent
CLUSTERS_PARQUET = ROOT_DIR / "data" / "processed" / "driving_behavior_clusters.parquet"
CLUSTERS_CSV = ROOT_DIR / "data" / "processed" / "driving_behavior_clusters.csv"
KMEANS_FILE = ROOT_DIR / "outputs" / "models" / "kmeans_driver_profile.joblib"
SCALER_FILE = ROOT_DIR / "outputs" / "models" / "scaler_driver_profile.joblib"
PCA_FILE = ROOT_DIR / "outputs" / "models" / "pca_driver_profile.joblib"

FIG_ELBOW = ROOT_DIR / "outputs" / "figures" / "elbow_and_silhouette_analysis.png"
FIG_PCA = ROOT_DIR / "outputs" / "figures" / "driving_clusters_pca_2d.png"
FIG_PROFILES = ROOT_DIR / "outputs" / "figures" / "cluster_profiles_comparison.png"


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
        size_kb = f.stat().st_size / 1024
        print(f"  [OK] {name}: Encontrado! ({size_kb:.1f} KB)")

    # 2. Carregar dataset rotulado
    print("\n[2/4] Validando distribuicao dos clusters na base rotulada...")
    df = pd.read_parquet(CLUSTERS_PARQUET)
    assert len(df) == 2481, f"Esperado 2.481 trechos, obteve {len(df)}"
    assert "cluster_label" in df.columns, "Coluna cluster_label ausente!"
    assert df["cluster_label"].isna().sum() == 0, "Existem valores nulos nos clusters!"

    counts = df["cluster_label"].value_counts()
    print("  [OK] Distribuicao confirmada:")
    for label, count in counts.items():
        print(f"       • {label:<22}: {count:>5,d} trechos ({count/len(df)*100:.1f}%)")

    # 3. Carregar os modelos treinados
    print("\n[3/4] Carregando modelos serializados de Machine Learning...")
    kmeans = joblib.load(KMEANS_FILE)
    scaler = joblib.load(SCALER_FILE)
    pca = joblib.load(PCA_FILE)
    print(f"  [OK] Modelo K-Means carregado com {kmeans.n_clusters} clusters.")
    print(f"  [OK] Normalizador StandardScaler carregado com {len(scaler.mean_)} variaveis.")
    print(f"  [OK] Modelo PCA carregado (Variancia explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%).")

    # 4. Teste de Predição em Tempo Real em Cenários Hipotéticos
    print("\n[4/4] Testando generalizacao do modelo com 3 novos motoristas hipoteticos:")
    label_names = {0: "Econômico / Suave", 1: "Moderado / Regular", 2: "Agressivo / Dinâmico"}

    feature_cols = [
        "mean_speed_kmh",
        "std_speed_kmh",
        "mean_pos_accel_ms2",
        "max_pos_accel_ms2",
        "hard_braking_rate_min",
        "rapid_accel_rate_min",
    ]

    scenarios = [
        {
            "nome": "Cenário 1: Condutor Calmo em Rodovia",
            "dados": [48.0, 10.5, 0.28, 0.75, 0.10, 0.00],
            "esperado": "Econômico / Suave",
        },
        {
            "nome": "Cenário 2: Condutor Padrao em Trânsito Misto",
            "dados": [31.0, 17.0, 0.62, 1.60, 0.35, 0.05],
            "esperado": "Moderado / Regular",
        },
        {
            "nome": "Cenário 3: Condutor Apressado com Freadas Fortes",
            "dados": [42.0, 22.0, 0.85, 2.40, 1.40, 1.20],
            "esperado": "Agressivo / Dinâmico",
        },
    ]

    for sc in scenarios:
        x_raw = np.array([sc["dados"]])
        x_scaled = scaler.transform(x_raw)
        pred_id = int(kmeans.predict(x_scaled)[0])
        pred_label = label_names[pred_id]

        print(f"\n  >>> {sc['nome']}")
        print(f"      Entradas: Vel.Media={sc['dados'][0]} km/h | Freadas/min={sc['dados'][4]} | Arrancadas/min={sc['dados'][5]}")
        print(f"      Predicao da IA: [{pred_label.upper()}]")
        assert pred_label == sc["esperado"], f"Falha no cenário {sc['nome']}! Esperado {sc['esperado']}, obteve {pred_label}"
        print("      Status: [VALIDADO COM SUCESSO]")

    print("\n" + "=" * 70)
    print("[SUCESSO] ETAPA 1.3 100% VALIDADA!")
    print("O modelo de IA de classificacao de perfis esta treinado, avaliado e salvo.")
    print("=" * 70)


if __name__ == "__main__":
    test_step_1_3()
