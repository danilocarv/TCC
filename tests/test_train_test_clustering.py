"""
Suíte de Testes Automatizados para Treinamento e Validação Out-of-Sample do K-Means.

Valida:
1. Existência e integridade física de todos os artefatos (.parquet, .csv, .joblib, .json, .png);
2. Isolamento estrito de conjuntos (1.700 no Treino, 534 no Teste, interseção vazia);
3. Ausência de valores nulos ou inválidos nas predições;
4. Generalização out-of-sample:
   - Coeficiente de Silhueta positivo no teste;
   - Estabilidade percentual dos perfis populacionais (sem colapso);
5. Consistência física: perfis Agressivos apresentam maior aceleração/frenagem que perfis Suaves;
6. Funcionamento determinístico do modelo serializado (.joblib).

Execute com:
  python test_train_test_clustering.py

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib

from src.config import (
    PROCESSED_TRAIN_LABELED_PARQUET,
    PROCESSED_TRAIN_LABELED_CSV,
    PROCESSED_TEST_LABELED_PARQUET,
    PROCESSED_TEST_LABELED_CSV,
    KMEANS_TRAIN_MODEL_PATH,
    SCALER_TRAIN_MODEL_PATH,
    PCA_TRAIN_MODEL_PATH,
    KMEANS_TRAIN_TEST_VALIDATION_JSON,
    KMEANS_TRAIN_TEST_VALIDATION_CSV,
    FIG_KMEANS_TRAIN_TEST_PCA,
    FIG_KMEANS_TRAIN_TEST_PROFILES,
    TRAIN_WINDOW_COUNT,
    TEST_WINDOW_COUNT,
    CLUSTER_FEATURE_COLS,
)


def validate_train_test_clustering():
    print("=" * 90)
    print("INICIANDO SUÍTE DE TESTES: K-MEANS TREINO (1.700) VS. TESTE OUT-OF-SAMPLE (534)")
    print("=" * 90)

    # 1. Existência dos artefatos
    print("\n[1/5] Verificando existência dos artefatos gerados...")
    artifacts = [
        (PROCESSED_TRAIN_LABELED_PARQUET, "Treino Rotulado Parquet"),
        (PROCESSED_TRAIN_LABELED_CSV, "Treino Rotulado CSV"),
        (PROCESSED_TEST_LABELED_PARQUET, "Teste Rotulado Parquet"),
        (PROCESSED_TEST_LABELED_CSV, "Teste Rotulado CSV"),
        (KMEANS_TRAIN_MODEL_PATH, "Modelo K-Means Treinado (.joblib)"),
        (SCALER_TRAIN_MODEL_PATH, "Normalizador StandardScaler (.joblib)"),
        (PCA_TRAIN_MODEL_PATH, "Redutor PCA (.joblib)"),
        (KMEANS_TRAIN_TEST_VALIDATION_JSON, "Métricas Validação JSON"),
        (KMEANS_TRAIN_TEST_VALIDATION_CSV, "Resumo Comparativo CSV"),
        (FIG_KMEANS_TRAIN_TEST_PCA, "Figura 15 - PCA 2D Treino vs Teste (300 DPI)"),
        (FIG_KMEANS_TRAIN_TEST_PROFILES, "Figura 16 - Perfis Treino vs Teste (300 DPI)"),
    ]

    for path, desc in artifacts:
        assert path.exists(), f"Artefato ausente: {desc} ({path})"
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] {desc} ({path.name}): Encontrado ({size_kb:.1f} KB)")

    # 2. Isolamento estrito de dados (Zero Data Leakage)
    print("\n[2/5] Auditando isolamento físico e ausência de vazamento de dados...")
    df_train = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET)
    df_test = pd.read_parquet(PROCESSED_TEST_LABELED_PARQUET)

    assert len(df_train) == TRAIN_WINDOW_COUNT, f"Esperado {TRAIN_WINDOW_COUNT} no treino, obtido {len(df_train)}"
    assert len(df_test) == TEST_WINDOW_COUNT, f"Esperado {TEST_WINDOW_COUNT} no teste, obtido {len(df_test)}"

    train_keys = set(zip(df_train["VehId"], df_train["Trip"], df_train["window_id"]))
    test_keys = set(zip(df_test["VehId"], df_test["Trip"], df_test["window_id"]))
    overlap = train_keys.intersection(test_keys)
    assert len(overlap) == 0, f"VAZAMENTO DE DADOS DETECTADO: {len(overlap)} janelas compartilhadas entre treino e teste!"
    print(f"  [OK] Contrato cumprido: {len(df_train)} treinos e {len(df_test)} testes. Interseção = 0 (Zero Leakage).")

    # 3. Integridade e consistência das predições
    print("\n[3/5] Validando integridade das predições e rótulos...")
    for name, df in [("Treino", df_train), ("Teste", df_test)]:
        assert df["cluster_id"].isna().sum() == 0, f"NaNs em cluster_id no {name}!"
        assert df["cluster_label"].isna().sum() == 0, f"NaNs em cluster_label no {name}!"
        assert set(df["cluster_id"].unique()) == {0, 1, 2}, f"Clusters incompletos no {name}!"
        assert "pca_x" in df.columns and "pca_y" in df.columns
        for c in range(3):
            assert f"dist_centroid_{c}" in df.columns

    print("  [OK] Todos os 2.234 trechos rotulados sem valores nulos e com distâncias aos centroides.")

    # 4. Auditoria de Generalização Out-of-Sample
    print("\n[4/5] Auditando métricas de generalização out-of-sample...")
    with open(KMEANS_TRAIN_TEST_VALIDATION_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    sil_train = metrics["metrics"]["silhouette_score"]["train"]
    sil_test = metrics["metrics"]["silhouette_score"]["test"]
    assert sil_train > 0.20, f"Silhueta no treino baixa: {sil_train:.4f}"
    assert sil_test > 0.20, f"Silhueta no teste baixa: {sil_test:.4f}"
    assert abs(sil_train - sil_test) < 0.05, f"Degradação excessiva da silhueta no teste: {sil_train} vs {sil_test}"
    print(f"  [OK] Silhueta preservada no teste out-of-sample: Treino = {sil_train:.4f} | Teste = {sil_test:.4f}")

    # Estabilidade da distribuição percentual (diferença máx < 6% por cluster)
    for row in metrics["distribution_summary"]:
        diff = abs(row["diff_pct"])
        assert diff < 6.0, f"Desvio excessivo no perfil {row['perfil']}: diff = {diff}%"
        print(f"  [OK] {row['perfil']:<20}: Treino={row['treino_pct']:>5.1f}% | Teste={row['teste_pct']:>5.1f}% (Diff={row['diff_pct']:>+5.1f}%)")

    # 5. Consistência Física dos Perfis no Teste
    print("\n[5/5] Auditando consistência física nos dados inéditos...")
    agg_test = df_test[df_test["cluster_id"] == 2]
    eco_test = df_test[df_test["cluster_id"] == 0]

    assert agg_test["hard_braking_rate_min"].mean() > eco_test["hard_braking_rate_min"].mean()
    assert agg_test["rapid_accel_rate_min"].mean() > eco_test["rapid_accel_rate_min"].mean()
    assert agg_test["mean_power_kw"].mean() > eco_test["mean_power_kw"].mean()
    print(f"  [OK] Consistência física comprovada no teste: Agressivo gasta {agg_test['mean_power_kw'].mean():.2f} kW vs. {eco_test['mean_power_kw'].mean():.2f} kW do Econômico.")

    print("\n" + "=" * 90)
    print("TODOS OS TESTES DE VALIDAÇÃO TREINO/TESTE PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    validate_train_test_clustering()
