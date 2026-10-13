"""
Script de Teste e Validação da Etapa 1.2.
Verifica:
1. A base regularizada em 1 Hz (ev_trips_1hz.parquet);
2. Os atributos comportamentais e novas features derivadas (Haversine e Tortuosidade);
3. A exportação individual das 2.234 janelas ativas em .csv e .parquet;
4. O embaralhamento reprodutível e a divisão física estrita em 1.700 Treino e 534 Teste.
Execute no terminal com: python test_step1_2.py
"""

import sys
from pathlib import Path

# Configura stdout para UTF-8 caso o terminal do Windows use cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from src.config import (
    PROCESSED_1HZ_PARQUET,
    PROCESSED_BEHAVIOR_PARQUET,
    PROCESSED_BEHAVIOR_CSV,
    PROCESSED_BEHAVIOR_METADATA_JSON,
    WINDOWS_TRAIN_CSV_DIR,
    WINDOWS_TRAIN_PARQUET_DIR,
    WINDOWS_TEST_CSV_DIR,
    WINDOWS_TEST_PARQUET_DIR,
    WINDOW_SPLIT_MANIFEST_CSV,
    PROCESSED_TRAIN_FEATURES_PARQUET,
    PROCESSED_TEST_FEATURES_PARQUET,
    TRAIN_WINDOW_COUNT,
    TEST_WINDOW_COUNT,
)


def test_step_1_2():
    print("=" * 72)
    print("INICIANDO TESTE DE VALIDACAO DA ETAPA 1.2 (JANELAS, SPLIT 1700/534 E FEATURES)")
    print("=" * 72)

    # 1. Verificar existência dos arquivos consolidados e manifesto
    print("\n[1/5] Verificando se os artefatos processados existem no disco...")
    for file_path, name in [
        (PROCESSED_1HZ_PARQUET, "Telemetria Regularizada 1 Hz (Parquet)"),
        (PROCESSED_BEHAVIOR_PARQUET, "Features Comportamentais Consolidado (Parquet)"),
        (PROCESSED_BEHAVIOR_CSV, "Features Comportamentais Consolidado (CSV)"),
        (PROCESSED_BEHAVIOR_METADATA_JSON, "Metadados JSON"),
        (WINDOW_SPLIT_MANIFEST_CSV, "Manifesto de Divisao Treino/Teste (CSV)"),
        (PROCESSED_TRAIN_FEATURES_PARQUET, "Features de Treino 1.700 (Parquet)"),
        (PROCESSED_TEST_FEATURES_PARQUET, "Features de Teste 534 (Parquet)"),
    ]:
        if not file_path.exists():
            print(f"[ERRO] Arquivo {name} nao encontrado em: {file_path}")
            sys.exit(1)
        size_kb = file_path.stat().st_size / 1024
        print(f"  [OK] {name}: Encontrado! ({size_kb:.1f} KB - {file_path.name})")

    # 2. Verificar contagem física dos 2.234 arquivos individuais (.csv e .parquet)
    print("\n[2/5] Validando contagem fisica dos arquivos individuais por janela (120s)...")
    train_csv_files = sorted(WINDOWS_TRAIN_CSV_DIR.glob("*.csv"))
    train_pq_files = sorted(WINDOWS_TRAIN_PARQUET_DIR.glob("*.parquet"))
    test_csv_files = sorted(WINDOWS_TEST_CSV_DIR.glob("*.csv"))
    test_pq_files = sorted(WINDOWS_TEST_PARQUET_DIR.glob("*.parquet"))

    print(f"  • Treino (.csv):     {len(train_csv_files):,d} arquivos em data/windows/train/csv/")
    print(f"  • Treino (.parquet): {len(train_pq_files):,d} arquivos em data/windows/train/parquet/")
    print(f"  • Teste  (.csv):     {len(test_csv_files):,d} arquivos em data/windows/test/csv/")
    print(f"  • Teste  (.parquet): {len(test_pq_files):,d} arquivos em data/windows/test/parquet/")

    assert len(train_csv_files) == TRAIN_WINDOW_COUNT, f"Esperado {TRAIN_WINDOW_COUNT} CSVs de treino!"
    assert len(train_pq_files) == TRAIN_WINDOW_COUNT, f"Esperado {TRAIN_WINDOW_COUNT} Parquets de treino!"
    assert len(test_csv_files) == TEST_WINDOW_COUNT, f"Esperado {TEST_WINDOW_COUNT} CSVs de teste!"
    assert len(test_pq_files) == TEST_WINDOW_COUNT, f"Esperado {TEST_WINDOW_COUNT} Parquets de teste!"

    # Validar ausência de interseção entre chaves de treino e teste
    df_manifest = pd.read_csv(WINDOW_SPLIT_MANIFEST_CSV)
    assert len(df_manifest) == TRAIN_WINDOW_COUNT + TEST_WINDOW_COUNT == 2234
    train_keys = set(
        zip(
            df_manifest[df_manifest["split"] == "train"]["VehId"],
            df_manifest[df_manifest["split"] == "train"]["Trip"],
            df_manifest[df_manifest["split"] == "train"]["window_id"],
        )
    )
    test_keys = set(
        zip(
            df_manifest[df_manifest["split"] == "test"]["VehId"],
            df_manifest[df_manifest["split"] == "test"]["Trip"],
            df_manifest[df_manifest["split"] == "test"]["window_id"],
        )
    )
    assert len(train_keys) == 1700 and len(test_keys) == 534
    assert train_keys.isdisjoint(test_keys), "Vazamento detectado entre Treino e Teste!"
    print("  [OK] Divisao 1.700 Treino / 534 Teste estritamente disjunta (zero vazamento)!")

    # 3. Carregar datasets e validar dimensões e novas features derivadas
    print("\n[3/5] Carregando datasets e validando novas features derivadas...")
    df = pd.read_parquet(PROCESSED_BEHAVIOR_PARQUET)
    df_train = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    df_test = pd.read_parquet(PROCESSED_TEST_FEATURES_PARQUET)

    assert len(df) == 2481, f"Esperado 2.481 janelas totais, mas obteve {len(df)}"
    assert len(df_train) == 1700, f"Esperado 1.700 janelas de treino, mas obteve {len(df_train)}"
    assert len(df_test) == 534, f"Esperado 534 janelas de teste, mas obteve {len(df_test)}"
    assert df.isna().sum().sum() == 0, "Existem valores nulos nos atributos calculados!"

    required_derived_cols = [
        "distance_total_m",
        "distance_straight_m",
        "tortuosity_index",
        "mean_neg_accel_ms2",
        "std_jerk_ms3",
        "high_jerk_rate_min",
        "idle_ratio",
    ]
    for col in required_derived_cols:
        assert col in df.columns, f"Coluna derivada ausente: {col}"

    print(f"  [OK] Todas as {len(required_derived_cols)} novas features derivadas presentes e sem NaNs.")

    # 4. Validar consistência física dos atributos
    print("\n[4/5] Validando limites e consistencia fisica das variaveis...")
    assert (df["mean_speed_kmh"] >= 0).all(), "Velocidade media negativa encontrada!"
    assert (df["max_pos_accel_ms2"] <= 6.01).all(), "Aceleracao positiva acima do limite fisico!"
    assert (df["min_neg_accel_ms2"] >= -8.01).all(), "Desaceleracao abaixo do limite fisico!"
    assert (df["distance_total_m"] >= 0).all(), "Distancia total negativa!"
    assert (df["distance_straight_m"] >= 0).all(), "Distancia Haversine negativa!"
    print("  [OK] Todos os atributos cinemáticos e espaciais estao dentro dos limites fisicos.")

    # 5. Inspecionar amostra individual de arquivo .parquet/.csv
    print("\n[5/5] Testando leitura direta do 1º arquivo embaralhado de Treino e de Teste:")
    sample_train_df = pd.read_parquet(train_pq_files[0])
    sample_test_df = pd.read_parquet(test_pq_files[0])
    print(f"  • Arquivo Treino #0: {train_pq_files[0].name} ({len(sample_train_df)} segundos a 1 Hz)")
    print(f"  • Arquivo Teste  #0: {test_pq_files[0].name} ({len(sample_test_df)} segundos a 1 Hz)")

    print("\n" + "=" * 72)
    print("[SUCESSO] ETAPA 1.2 VALIDADA COM DIVISAO 1.700 TREINO / 534 TESTE!")
    print("=" * 72)


if __name__ == "__main__":
    test_step_1_2()
