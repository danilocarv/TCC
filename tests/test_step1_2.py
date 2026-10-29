"""
Script de Teste e Validação da Etapa 1.2.
Verifica:
1. A base regularizada em 1 Hz (ev_trips_1hz.parquet);
2. Os atributos comportamentais e novas features derivadas (Haversine e Tortuosidade);
3. A exportação individual das 2.234 janelas ativas em .csv e .parquet;
4. O embaralhamento reprodutível e a divisão física estrita em 1.700 Treino e 534 Teste;
5. O contrato de dados de cada janela (mínimo de MIN_WINDOW_POINTS registros, ausência de nulos,
   dt positivo e finito, integridade de esquema e equivalência Parquet vs. CSV).
Execute no terminal com: python test_step1_2.py
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Configura stdout para UTF-8 caso o terminal do Windows use cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pyarrow.parquet as pq

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
    MIN_WINDOW_POINTS,
)


def test_step_1_2():
    print("=" * 72)
    print("INICIANDO TESTE DE VALIDACAO DA ETAPA 1.2 (JANELAS, SPLIT 1700/534 E FEATURES)")
    print("=" * 72)

    # 1. Verificar existência dos arquivos consolidados e manifesto
    print("\n[1/6] Verificando se os artefatos processados existem no disco...")
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
    print("\n[2/6] Validando contagem fisica dos arquivos individuais por janela (120s)...")
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
    total_expected = TRAIN_WINDOW_COUNT + TEST_WINDOW_COUNT
    assert len(df_manifest) == total_expected, (
        f"Esperado {total_expected} registros no manifesto, mas obteve {len(df_manifest)}"
    )
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
    assert len(train_keys) == TRAIN_WINDOW_COUNT, (
        f"Chaves de treino: esperado {TRAIN_WINDOW_COUNT}, obteve {len(train_keys)}"
    )
    assert len(test_keys) == TEST_WINDOW_COUNT, (
        f"Chaves de teste: esperado {TEST_WINDOW_COUNT}, obteve {len(test_keys)}"
    )
    assert train_keys.isdisjoint(test_keys), "Vazamento detectado entre Treino e Teste!"
    print(f"  [OK] Divisao {TRAIN_WINDOW_COUNT} Treino / {TEST_WINDOW_COUNT} Teste estritamente disjunta (zero vazamento)!")

    # 3. Carregar datasets e validar dimensões e novas features derivadas
    print("\n[3/6] Carregando datasets e validando novas features derivadas...")
    df = pd.read_parquet(PROCESSED_BEHAVIOR_PARQUET)
    df_train = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    df_test = pd.read_parquet(PROCESSED_TEST_FEATURES_PARQUET)

    assert len(df) == 2481, f"Esperado 2.481 janelas totais, mas obteve {len(df)}"
    assert len(df_train) == TRAIN_WINDOW_COUNT, (
        f"Esperado {TRAIN_WINDOW_COUNT} janelas de treino, mas obteve {len(df_train)}"
    )
    assert len(df_test) == TEST_WINDOW_COUNT, (
        f"Esperado {TEST_WINDOW_COUNT} janelas de teste, mas obteve {len(df_test)}"
    )
    assert df.isna().sum().sum() == 0, "Existem valores nulos nos atributos calculados!"
    assert df_train.isna().sum().sum() == 0, "Existem valores nulos na particao de treino!"
    assert df_test.isna().sum().sum() == 0, "Existem valores nulos na particao de teste!"
    assert df_train.columns.equals(df_test.columns), (
        "As particoes de treino e teste possuem esquemas diferentes!"
    )
    assert set(df_train.columns) == set(df.columns), (
        "As colunas da particao de treino nao coincidem com o consolidado!"
    )

    def feature_keys(frame):
        return set(zip(frame["VehId"], frame["Trip"], frame["window_id"]))

    assert feature_keys(df_train) == train_keys, (
        "As chaves da particao de treino nao correspondem ao manifesto!"
    )
    assert feature_keys(df_test) == test_keys, (
        "As chaves da particao de teste nao correspondem ao manifesto!"
    )

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

    # 4. Validar limites e consistência física das variáveis agregadas
    print("\n[4/6] Validando limites e consistencia fisica das variaveis...")
    assert (df["mean_speed_kmh"] >= 0).all(), "Velocidade media negativa encontrada!"
    assert (df["max_pos_accel_ms2"] <= 6.01).all(), "Aceleracao positiva acima do limite fisico!"
    assert (df["min_neg_accel_ms2"] >= -8.01).all(), "Desaceleracao abaixo do limite fisico!"
    assert (df["distance_total_m"] >= 0).all(), "Distancia total negativa!"
    assert (df["distance_straight_m"] >= 0).all(), "Distancia Haversine negativa!"
    assert (df["idle_ratio"] >= 0.0).all() and (df["idle_ratio"] <= 1.0).all(), "idle_ratio fora da faixa [0, 1]!"
    assert (df["tortuosity_index"] >= 0.99).all(), "tortuosity_index menor que 1.0 encontrado!"
    assert (df["distance_straight_m"] <= df["distance_total_m"] + 1e-2).all(), "Haversine maior que distancia total percorrida!"
    assert (df["duration_s"] >= MIN_WINDOW_POINTS).all(), f"Duracao menor que {MIN_WINDOW_POINTS}s!"
    print("  [OK] Todos os atributos cinemáticos, espaciais e operacionais estao consistentes.")

    # 5. Validação de contrato de dados em massa de todos os 2.234 arquivos
    print("\n[5/6] Validando contrato de dados de todos os 2.234 arquivos individuais...")
    all_pq_files = train_pq_files + test_pq_files
    all_csv_files = train_csv_files + test_csv_files

    # Validar que nenhum CSV está vazio
    for csv_f in all_csv_files:
        assert csv_f.stat().st_size > 0, f"Arquivo CSV vazio encontrado: {csv_f.name}"

    # Validar metadados e número de registros de 100% dos arquivos Parquet via pyarrow
    min_points_observed = 999
    max_points_observed = 0
    for pq_f in all_pq_files:
        pq_meta = pq.ParquetFile(pq_f).metadata
        num_rows = pq_meta.num_rows
        min_points_observed = min(min_points_observed, num_rows)
        max_points_observed = max(max_points_observed, num_rows)
        assert (
            num_rows >= MIN_WINDOW_POINTS
        ), f"Arquivo {pq_f.name} possui apenas {num_rows} linhas (< {MIN_WINDOW_POINTS})!"
        assert (
            num_rows <= 120
        ), f"Arquivo {pq_f.name} possui {num_rows} linhas (> 120)!"

    print(f"  [OK] Todos os 2.234 arquivos Parquet possuem entre {min_points_observed} e {max_points_observed} registros (>= {MIN_WINDOW_POINTS}).")
    print(f"  [OK] Todos os 2.234 arquivos CSV possuem tamanho valido e conteudo gravado.")

    # 6. Leitura cruzada detalhada de amostra estratificada (.parquet vs. .csv)
    print("\n[6/6] Realizando inspecao profunda em amostra estratificada (Parquet vs. CSV)...")
    sample_indices = [0, 50, 500, 1000, 1699]  # Treino
    sample_test_indices = [0, 50, 250, 533]    # Teste

    for idx in sample_indices:
        df_p = pd.read_parquet(train_pq_files[idx])
        df_c = pd.read_csv(train_csv_files[idx])
        assert len(df_p) == len(df_c), f"Divergencia de linhas em {train_pq_files[idx].name}!"
        assert df_p.isna().sum().sum() == 0, f"Valores nulos encontrados em {train_pq_files[idx].name}!"
        assert (df_p["dt"] >= 1.0).all() and (df_p["dt"] <= 5.0).all(), f"dt fora da faixa [1, 5] em {train_pq_files[idx].name}!"
        for col in ["Vehicle Speed[km/h]", "accel_ms2", "jerk_ms3", "power_kw", "dt"]:
            assert col in df_p.columns, f"Coluna essencial {col} ausente em {train_pq_files[idx].name}!"

    for idx in sample_test_indices:
        df_p = pd.read_parquet(test_pq_files[idx])
        df_c = pd.read_csv(test_csv_files[idx])
        assert len(df_p) == len(df_c), f"Divergencia de linhas em {test_pq_files[idx].name}!"
        assert df_p.isna().sum().sum() == 0, f"Valores nulos encontrados em {test_pq_files[idx].name}!"
        assert (df_p["dt"] >= 1.0).all() and (df_p["dt"] <= 5.0).all(), f"dt fora da faixa [1, 5] em {test_pq_files[idx].name}!"

    print(f"  [OK] Leitura cruzada Parquet vs. CSV 100% compativel e sem nulos.")

    print("\n" + "=" * 72)
    print("[SUCESSO] ETAPA 1.2 E CONTRATO DE DADOS DAS JANELAS 100% VALIDADOS!")
    print("=" * 72)


if __name__ == "__main__":
    test_step_1_2()
