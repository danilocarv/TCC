"""
Módulo de Engenharia de Atributos Comportamentais de Condução e Divisão de Janelas (Etapa 1.2).
1. Regulariza a telemetria bruta para 1 Hz exato (ev_trips_1hz.parquet).
2. Segmenta as viagens em janelas temporais de 120 segundos e extrai features cinemáticas
   e features derivadas espaciais (Distância Total, Haversine em linha reta, Tortuosidade).
3. Isola as 2.234 janelas ativas (v_media >= 8.0 km/h), aplica embaralhamento reprodutível
   (Shuffle com random_state=42) e exporta cada janela individualmente em .csv e .parquet
   divididas fisicamente em Treino (1.700 arquivos) e Teste (534 arquivos).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import os
import sys
import json
import time
import math
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.config import (
    RAW_EV_PARQUET,
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
    PROCESSED_TRAIN_FEATURES_CSV,
    PROCESSED_TEST_FEATURES_PARQUET,
    PROCESSED_TEST_FEATURES_CSV,
    WINDOW_SIZE_SECONDS,
    MIN_WINDOW_POINTS,
    ACTIVE_SPEED_THRESHOLD_KMH,
    TRAIN_WINDOW_COUNT,
    TEST_WINDOW_COUNT,
    RANDOM_SEED,
    HARD_BRAKING_THRESHOLD,
    RAPID_ACCEL_THRESHOLD,
    HIGH_JERK_THRESHOLD,
    ensure_directories,
)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância geodésica em linha reta (metros) entre duas coordenadas GPS
    (em graus decimais) utilizando a Fórmula de Haversine.
    """
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return 0.0
    r_earth_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2 - lat1))
    delta_lambda = math.radians(float(lon2 - lon1))

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(r_earth_m * c)


def compute_point_kinematics(df_telemetry: pd.DataFrame, save_1hz: bool = True) -> pd.DataFrame:
    """
    Calcula variáveis cinemáticas ponto a ponto regularizadas em grade temporal de 1 Hz exato.
    Remove os ruídos de polling sub-segundo do barramento CAN (0.1s a 0.4s) que inflacionavam
    artificialmente a aceleração (dv/dt) em leituras repetidas dos sensores OBD-II.
    Salva o artefato intermediário em data/processed/ev_trips_1hz.parquet.
    """
    print("-> [1/5] Regularizando telemetria para 1 Hz exato e calculando cinemática...")
    start_t = time.time()
    ensure_directories()

    df = df_telemetry.copy()

    # 1. Converter timestamp relativo de cada viagem em segundos inteiros exatos (1 Hz)
    df["second_id"] = (
        (df["Timestamp(ms)"] - df.groupby(["VehId", "Trip"])["Timestamp(ms)"].transform("min")) / 1000.0
    ).round().astype(int)

    # 2. Reamostragem/agregação para 1 segundo exato por viagem
    # Agrupa múltiplas mensagens da rede CAN que chegam no mesmo segundo
    df_1hz = df.groupby(["VehId", "Trip", "second_id"], as_index=False).agg({
        "DayNum": "first",
        "Latitude[deg]": "first",
        "Longitude[deg]": "first",
        "Vehicle Speed[km/h]": "mean",
        "OAT[DegC]": "mean",
        "Air Conditioning Power[Watts]": "mean",
        "Heater Power[Watts]": "mean",
        "HV Battery Current[A]": "mean",
        "HV Battery SOC[%]": "last",
        "HV Battery Voltage[V]": "mean",
    })

    # Garantir ordenação temporal estrita
    df_1hz.sort_values(by=["VehId", "Trip", "second_id"], inplace=True)
    df_1hz.reset_index(drop=True, inplace=True)

    # Tempo relativo decorrido e ID da janela de 120s dentro de cada viagem
    df_1hz["elapsed_s"] = df_1hz.groupby(["VehId", "Trip"])["second_id"].transform(
        lambda x: x - x.min()
    )
    df_1hz["window_id"] = (df_1hz["elapsed_s"] // WINDOW_SIZE_SECONDS).astype(int)

    # Delta de tempo exato em segundos entre registros (1.0s)
    df_1hz["dt"] = (
        df_1hz.groupby(["VehId", "Trip"])["second_id"].diff().fillna(1.0).clip(lower=1.0, upper=5.0)
    )

    # Conversão de velocidade: km/h para m/s
    df_1hz["v_ms"] = df_1hz["Vehicle Speed[km/h]"] / 3.6

    # Suavização móvel de 3 pontos para velocidade (filtra ruído de degrau de 1 km/h do OBD)
    df_1hz["v_smooth_ms"] = (
        df_1hz.groupby(["VehId", "Trip"])["v_ms"]
        .transform(lambda s: s.rolling(window=3, min_periods=1, center=True).mean())
    )

    # Aceleração longitudinal real em 1 Hz: a = dv / dt (m/s²)
    df_1hz["accel_ms2"] = (
        df_1hz.groupby(["VehId", "Trip"])["v_smooth_ms"].diff() / df_1hz["dt"]
    ).fillna(0.0).clip(lower=-8.0, upper=6.0)

    # Jerk (taxa de variação da aceleração em m/s³)
    df_1hz["jerk_ms3"] = (
        df_1hz.groupby(["VehId", "Trip"])["accel_ms2"].diff() / df_1hz["dt"]
    ).fillna(0.0).clip(lower=-15.0, upper=15.0)

    # Potência instantânea consumida da bateria em kW (P = - V * I / 1000)
    # Valores positivos = consumo; valores negativos = regeneração
    df_1hz["power_kw"] = -(df_1hz["HV Battery Voltage[V]"] * df_1hz["HV Battery Current[A]"]) / 1000.0

    if save_1hz:
        df_1hz.to_parquet(PROCESSED_1HZ_PARQUET, index=False, engine="pyarrow", compression="snappy")
        size_mb = os.path.getsize(PROCESSED_1HZ_PARQUET) / (1024 * 1024)
        print(f"   Base 1 Hz salva em: {PROCESSED_1HZ_PARQUET.name} ({size_mb:.2f} MB)")

    elapsed = time.time() - start_t
    print(f"   Telemetria regularizada a 1 Hz ({len(df_1hz):,d} segundos de operação) em {elapsed:.1f}s!")
    return df_1hz


def compute_single_window_features(
    group: pd.DataFrame,
    veh_id: int,
    trip_id: int,
    win_id: int,
) -> dict:
    """
    Calcula o vetor resumo de uma única janela de 120 segundos (6 features cinemáticas
    principais do K-Means + novas features derivadas espaciais e dinâmicas).
    """
    duration_s = float(group["dt"].sum())
    speeds = group["Vehicle Speed[km/h]"]
    v_ms = group["v_ms"]
    accels = group["accel_ms2"]
    jerks = group["jerk_ms3"]
    powers = group["power_kw"]

    # Acelerações positivas (quando o motorista está acelerando)
    pos_accels = accels[accels > 0.1]
    mean_pos_accel = float(pos_accels.mean()) if len(pos_accels) > 0 else 0.0
    max_pos_accel = float(pos_accels.max()) if len(pos_accels) > 0 else 0.0

    # Desacelerações / Frenagens (quando o motorista está freando)
    neg_accels = accels[accels < -0.1]
    mean_neg_accel = float(neg_accels.mean()) if len(neg_accels) > 0 else 0.0
    min_neg_accel = float(neg_accels.min()) if len(neg_accels) > 0 else 0.0

    # Estabilidade de Jerk
    std_jerk = float(jerks.std()) if len(jerks) > 1 else 0.0

    # Eventos extremos comportamentais
    hard_brakes = int((accels <= HARD_BRAKING_THRESHOLD).sum())
    rapid_accels = int((accels >= RAPID_ACCEL_THRESHOLD).sum())
    high_jerks = int((jerks.abs() >= HIGH_JERK_THRESHOLD).sum())

    # Taxas normalizadas por minuto
    duration_min = max(duration_s / 60.0, 0.1)
    hard_brake_rate = float(hard_brakes / duration_min)
    rapid_accel_rate = float(rapid_accels / duration_min)
    high_jerk_rate = float(high_jerks / duration_min)

    # Proporção de tempo parado/ocioso (velocidade < 2.0 km/h)
    idle_points = (speeds < 2.0).sum()
    idle_ratio = float(idle_points / len(group))

    # Novas Features Derivadas Espaciais (Distância Total, Haversine em Linha Reta e Tortuosidade)
    distance_total_m = float((v_ms * group["dt"]).sum())

    valid_gps = group.dropna(subset=["Latitude[deg]", "Longitude[deg]"])
    if len(valid_gps) >= 2:
        lat_start = float(valid_gps["Latitude[deg]"].iloc[0])
        lon_start = float(valid_gps["Longitude[deg]"].iloc[0])
        lat_end = float(valid_gps["Latitude[deg]"].iloc[-1])
        lon_end = float(valid_gps["Longitude[deg]"].iloc[-1])
        distance_straight_m = haversine_distance_m(lat_start, lon_start, lat_end, lon_end)
    else:
        distance_straight_m = 0.0

    # A distância em linha reta nunca pode exceder fisicamente a distância real percorrida
    # Em veículos parados ou com deslocamento ínfimo (< 1.0 m), neutraliza ruído de jitter do GPS
    if distance_total_m <= 1.0:
        distance_straight_m = 0.0
        tortuosity_index = 1.0
    else:
        if distance_straight_m > distance_total_m:
            distance_straight_m = distance_total_m
        tortuosity_index = float(distance_total_m / max(distance_straight_m, 1.0))

    # Energia consumida aproximada no trecho: E (kWh) = integral(P * dt) / 3600
    trecho_energy_kwh = float((powers * group["dt"]).sum() / 3600.0)

    return {
        "VehId": int(veh_id),
        "Trip": int(trip_id),
        "window_id": int(win_id),
        "duration_s": round(duration_s, 1),
        "sample_count": int(len(group)),
        # Atributos de Velocidade
        "mean_speed_kmh": round(float(speeds.mean()), 2),
        "max_speed_kmh": round(float(speeds.max()), 2),
        "std_speed_kmh": round(float(speeds.std()), 2) if len(speeds) > 1 else 0.0,
        # Atributos de Aceleração e Jerk
        "mean_pos_accel_ms2": round(mean_pos_accel, 3),
        "max_pos_accel_ms2": round(max_pos_accel, 3),
        "mean_neg_accel_ms2": round(mean_neg_accel, 3),
        "min_neg_accel_ms2": round(min_neg_accel, 3),
        "std_jerk_ms3": round(std_jerk, 3),
        # Eventos Bruscos e Estabilidade
        "hard_braking_events": hard_brakes,
        "hard_braking_rate_min": round(hard_brake_rate, 2),
        "rapid_accel_events": rapid_accels,
        "rapid_accel_rate_min": round(rapid_accel_rate, 2),
        "high_jerk_events": high_jerks,
        "high_jerk_rate_min": round(high_jerk_rate, 2),
        "idle_ratio": round(idle_ratio, 3),
        # Features Derivadas Espaciais (Geometria da Via)
        "distance_total_m": round(distance_total_m, 2),
        "distance_straight_m": round(distance_straight_m, 2),
        "tortuosity_index": round(tortuosity_index, 3),
        # Contexto Energético do Trecho
        "mean_power_kw": round(float(powers.mean()), 2),
        "total_energy_kwh": round(trecho_energy_kwh, 4),
    }


def extract_driving_behavior_windows(
    df_kinematics: pd.DataFrame,
    window_size_seconds: int = WINDOW_SIZE_SECONDS,
    min_points: int = MIN_WINDOW_POINTS,
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], pd.DataFrame]]:
    """
    Segmenta as viagens em janelas operacionais de condução (120 segundos),
    extrai os atributos comportamentais + features derivadas de cada trecho
    e retorna também o dicionário com os DataFrames individuais de cada janela.
    """
    print(f"-> [2/5] Segmentando viagens em janelas temporais de {window_size_seconds}s e extraindo features derivadas...")
    start_t = time.time()

    df = df_kinematics.copy()
    if "elapsed_s" not in df.columns:
        df["elapsed_s"] = df.groupby(["VehId", "Trip"])["second_id"].transform(
            lambda x: x - x.min()
        )
    if "window_id" not in df.columns:
        df["window_id"] = (df["elapsed_s"] // window_size_seconds).astype(int)

    feature_rows = []
    window_groups: dict[tuple[int, int, int], pd.DataFrame] = {}

    grouped = df.groupby(["VehId", "Trip", "window_id"])
    total_groups = len(grouped)
    print(f"   Total de janelas brutas identificadas: {total_groups:,d}")

    for (veh_id, trip_id, win_id), group in grouped:
        if len(group) < min_points:
            continue

        key = (int(veh_id), int(trip_id), int(win_id))
        window_groups[key] = group.reset_index(drop=True)
        row = compute_single_window_features(group, int(veh_id), int(trip_id), int(win_id))
        feature_rows.append(row)

    df_features = pd.DataFrame(feature_rows)
    df_features.fillna(0.0, inplace=True)

    elapsed = time.time() - start_t
    print(f"   Segmentação concluída em {elapsed:.1f}s!")
    print(f"   Total de janelas válidas (>= {min_points}s): {len(df_features):,d}")
    return df_features, window_groups


def export_and_split_active_windows(
    df_features: pd.DataFrame,
    window_groups: dict[tuple[int, int, int], pd.DataFrame],
    train_count: int = TRAIN_WINDOW_COUNT,
    test_count: int = TEST_WINDOW_COUNT,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Isola as 2.234 janelas de condução ativa (mean_speed_kmh >= 8.0 km/h),
    embaralha de forma reprodutível (random_state=42) e exporta cada janela
    individualmente em formato .csv e .parquet nas pastas de Treino (1.700) e Teste (534).
    """
    print(
        f"\n-> [3/5] Embaralhando as 2.234 janelas ativas (seed={random_state}) "
        f"e exportando arquivos individuais (.csv e .parquet)..."
    )
    start_t = time.time()
    ensure_directories()

    # Limpar eventuais arquivos antigos nas pastas de janelas
    for folder in [
        WINDOWS_TRAIN_CSV_DIR,
        WINDOWS_TRAIN_PARQUET_DIR,
        WINDOWS_TEST_CSV_DIR,
        WINDOWS_TEST_PARQUET_DIR,
    ]:
        for old_file in folder.glob("*.*"):
            old_file.unlink()

    df_all = df_features.copy()
    active_mask = df_all["mean_speed_kmh"] >= ACTIVE_SPEED_THRESHOLD_KMH
    df_active = df_all[active_mask].copy().reset_index(drop=True)

    expected_total_active = train_count + test_count
    if len(df_active) != expected_total_active:
        raise ValueError(
            f"Esperado {expected_total_active} janelas ativas (1700 + 534), mas foram encontradas {len(df_active)}."
        )

    # Embaralhamento reprodutível (Shuffle) das 2.234 janelas ativas
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(len(df_active))
    df_shuffled = df_active.iloc[shuffled_indices].copy().reset_index(drop=True)

    # Vetorização de split, split_rank e file_stem
    n_active = len(df_shuffled)
    shuffle_order = np.arange(n_active, dtype=int)
    split_labels = np.where(shuffle_order < train_count, "train", "test")
    split_ranks = np.where(shuffle_order < train_count, shuffle_order, shuffle_order - train_count)

    file_stems = [
        f"{s}_{r:04d}_veh{v}_trip{t}_w{w:03d}"
        for s, r, v, t, w in zip(
            split_labels,
            split_ranks,
            df_shuffled["VehId"],
            df_shuffled["Trip"],
            df_shuffled["window_id"],
        )
    ]

    df_shuffled["shuffle_order"] = shuffle_order
    df_shuffled["split"] = split_labels
    df_shuffled["split_rank"] = split_ranks
    df_shuffled["window_file"] = file_stems

    # Construção vetorizada do manifesto
    csv_rel_paths = [
        f"data/windows/{s}/csv/{stem}.csv" for s, stem in zip(split_labels, file_stems)
    ]
    parquet_rel_paths = [
        f"data/windows/{s}/parquet/{stem}.parquet" for s, stem in zip(split_labels, file_stems)
    ]

    df_manifest = pd.DataFrame({
        "shuffle_order": shuffle_order,
        "split": split_labels,
        "split_rank": split_ranks,
        "file_stem": file_stems,
        "VehId": df_shuffled["VehId"],
        "Trip": df_shuffled["Trip"],
        "window_id": df_shuffled["window_id"],
        "sample_count": df_shuffled["sample_count"],
        "duration_s": df_shuffled["duration_s"],
        "mean_speed_kmh": df_shuffled["mean_speed_kmh"],
        "distance_total_m": df_shuffled["distance_total_m"],
        "distance_straight_m": df_shuffled["distance_straight_m"],
        "tortuosity_index": df_shuffled["tortuosity_index"],
        "csv_rel_path": csv_rel_paths,
        "parquet_rel_path": parquet_rel_paths,
    })
    df_manifest.to_csv(WINDOW_SPLIT_MANIFEST_CSV, index=False)

    # Gravação física em disco (apenas I/O no loop)
    for s_lbl, stem, veh_id, trip_id, win_id in zip(
        split_labels,
        file_stems,
        df_shuffled["VehId"],
        df_shuffled["Trip"],
        df_shuffled["window_id"],
    ):
        csv_dir = WINDOWS_TRAIN_CSV_DIR if s_lbl == "train" else WINDOWS_TEST_CSV_DIR
        parquet_dir = WINDOWS_TRAIN_PARQUET_DIR if s_lbl == "train" else WINDOWS_TEST_PARQUET_DIR
        key = (int(veh_id), int(trip_id), int(win_id))

        df_window_ts = window_groups[key]
        df_window_ts.to_csv(csv_dir / f"{stem}.csv", index=False)
        df_window_ts.to_parquet(parquet_dir / f"{stem}.parquet", index=False, engine="pyarrow", compression="snappy")

    # Anotação vetorizada e segura em df_all via merge(validate="one_to_one")
    split_info = df_manifest[
        ["VehId", "Trip", "window_id", "shuffle_order", "split", "split_rank", "file_stem"]
    ].rename(columns={"file_stem": "window_file"})

    df_all = df_all.merge(
        split_info,
        on=["VehId", "Trip", "window_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    df_all["shuffle_order"] = df_all["shuffle_order"].fillna(-1).astype(int)
    df_all["split"] = df_all["split"].fillna("inactive")
    df_all["split_rank"] = df_all["split_rank"].fillna(-1).astype(int)
    df_all["window_file"] = df_all["window_file"].fillna("")

    df_train = df_shuffled[df_shuffled["split"] == "train"].copy().reset_index(drop=True)
    df_test = df_shuffled[df_shuffled["split"] == "test"].copy().reset_index(drop=True)

    # Salvar as tabelas de features separadas de Treino (1.700) e Teste (534)
    df_train.to_parquet(PROCESSED_TRAIN_FEATURES_PARQUET, index=False, engine="pyarrow", compression="snappy")
    df_train.to_csv(PROCESSED_TRAIN_FEATURES_CSV, index=False)
    df_test.to_parquet(PROCESSED_TEST_FEATURES_PARQUET, index=False, engine="pyarrow", compression="snappy")
    df_test.to_csv(PROCESSED_TEST_FEATURES_CSV, index=False)

    elapsed = time.time() - start_t
    print(f"   Exportação concluída em {elapsed:.1f}s!")
    print(f"   • Treino (data/windows/train/): {len(df_train):,d} arquivos .csv e {len(df_train):,d} arquivos .parquet")
    print(f"   • Teste  (data/windows/test/):  {len(df_test):,d} arquivos .csv e {len(df_test):,d} arquivos .parquet")
    print(f"   • Manifesto de divisão salvo em: {WINDOW_SPLIT_MANIFEST_CSV.name}")

    return df_all, df_train, df_test, df_manifest


def save_features_and_metadata(
    df_features: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    """Salva os atributos comportamentais extraídos em Parquet, CSV e JSON de metadados."""
    print(f"\n-> [4/5] Salvando dataset consolidado e metadados estatísticos...")
    ensure_directories()

    # 1. Salvar Parquet consolidado
    df_features.to_parquet(PROCESSED_BEHAVIOR_PARQUET, index=False, engine="pyarrow", compression="snappy")
    parquet_size_mb = os.path.getsize(PROCESSED_BEHAVIOR_PARQUET) / (1024 * 1024)
    print(f"   Arquivo Parquet gerado: {PROCESSED_BEHAVIOR_PARQUET.name} ({parquet_size_mb:.2f} MB)")

    # 2. Salvar CSV consolidado
    df_features.to_csv(PROCESSED_BEHAVIOR_CSV, index=False)
    csv_size_mb = os.path.getsize(PROCESSED_BEHAVIOR_CSV) / (1024 * 1024)
    print(f"   Arquivo CSV gerado: {PROCESSED_BEHAVIOR_CSV.name} ({csv_size_mb:.2f} MB)")

    # 3. Metadados e sumário estatístico
    feature_cols = [
        "mean_speed_kmh", "max_speed_kmh", "std_speed_kmh",
        "mean_pos_accel_ms2", "max_pos_accel_ms2", "mean_neg_accel_ms2", "min_neg_accel_ms2",
        "std_jerk_ms3",
        "hard_braking_events", "hard_braking_rate_min",
        "rapid_accel_events", "rapid_accel_rate_min",
        "high_jerk_events", "high_jerk_rate_min", "idle_ratio",
        "distance_total_m", "distance_straight_m", "tortuosity_index",
    ]

    stats_dict = {}
    for col in feature_cols:
        stats_dict[col] = {
            "mean": float(round(df_features[col].mean(), 3)),
            "std": float(round(df_features[col].std(), 3)),
            "min": float(round(df_features[col].min(), 3)),
            "p50": float(round(df_features[col].median(), 3)),
            "p90": float(round(df_features[col].quantile(0.90), 3)),
            "max": float(round(df_features[col].max(), 3)),
        }

    metadata = {
        "dataset": "VED - Pure Electric Vehicles Driving Behavior Features",
        "description": f"Atributos comportamentais e espaciais agregados em janelas de 120 segundos com divisão Treino ({len(df_train):,d}) / Teste ({len(df_test):,d})",
        "total_windows": int(len(df_features)),
        "active_windows_total": int(len(df_train) + len(df_test)),
        "train_windows_count": int(len(df_train)),
        "test_windows_count": int(len(df_test)),
        "inactive_windows_count": int((df_features["split"] == "inactive").sum()),
        "random_seed_shuffle": int(RANDOM_SEED),
        "total_trips_covered": int(df_features["Trip"].nunique()),
        "vehicles": [int(x) for x in sorted(df_features["VehId"].unique())],
        "feature_columns": feature_cols,
        "descriptive_statistics": stats_dict,
    }

    with open(PROCESSED_BEHAVIOR_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"   Metadados salvos em: {PROCESSED_BEHAVIOR_METADATA_JSON.name}")

    total_active = len(df_train) + len(df_test)
    train_pct = (len(df_train) / total_active * 100.0) if total_active > 0 else 0.0
    test_pct = (len(df_test) / total_active * 100.0) if total_active > 0 else 0.0

    print("\n-> [5/5] ============ RESUMO DOS ATRIBUTOS E DIVISÃO TREINO/TESTE ============")
    print(f"Total de Trechos (Janelas 120s): {len(df_features):,d} (Ativas: {total_active:,d})")
    print(f"  • Conjunto de Treino: {len(df_train):,d} janelas ({train_pct:.1f}%) -> data/windows/train/ (.csv e .parquet)")
    print(f"  • Conjunto de Teste:    {len(df_test):,d} janelas ({test_pct:.1f}%) -> data/windows/test/  (.csv e .parquet)")
    print("\nEstatísticas das Variáveis Cinemáticas e Novas Features Derivadas:")
    print(f"  • Velocidade Média:        {stats_dict['mean_speed_kmh']['mean']:.1f} km/h (Max: {stats_dict['mean_speed_kmh']['max']:.1f} km/h)")
    print(f"  • Desvio Padrão Velocidade:{stats_dict['std_speed_kmh']['mean']:.2f} km/h")
    print(f"  • Aceleração Média Pos.:   {stats_dict['mean_pos_accel_ms2']['mean']:.2f} m/s²")
    print(f"  • Frenagens Bruscas/min:   {stats_dict['hard_braking_rate_min']['mean']:.2f} eventos/min")
    print(f"  • Acelerações Bruscas/min: {stats_dict['rapid_accel_rate_min']['mean']:.2f} eventos/min")
    print(f"  • Trancos Severos/min:     {stats_dict['high_jerk_rate_min']['mean']:.2f} eventos/min")
    print(f"  • Distância Média/Janela:  {stats_dict['distance_total_m']['mean']:.1f} m (Reta Haversine: {stats_dict['distance_straight_m']['mean']:.1f} m)")
    print(f"  • Índice de Tortuosidade:  {stats_dict['tortuosity_index']['mean']:.2f} (Mediana: {stats_dict['tortuosity_index']['p50']:.2f})")
    print("==================================================================================\n")


def run_feature_pipeline():
    """Executa a pipeline completa da Etapa 1.2 (1 Hz, Features Derivadas, Shuffle e Split 1700/534)."""
    if not os.path.exists(RAW_EV_PARQUET):
        raise FileNotFoundError(
            f"Base de dados bruta não encontrada em: {RAW_EV_PARQUET}. Execute a Etapa 1.1 primeiro."
        )

    print("-> Carregando base de telemetria dos EVs...")
    df_telemetry = pd.read_parquet(RAW_EV_PARQUET)
    print(f"   Carregados {len(df_telemetry):,d} registros brutos.")

    df_kinematics = compute_point_kinematics(df_telemetry, save_1hz=True)
    df_features, window_groups = extract_driving_behavior_windows(df_kinematics)
    df_features, df_train, df_test, _ = export_and_split_active_windows(df_features, window_groups)
    save_features_and_metadata(df_features, df_train, df_test)


if __name__ == "__main__":
    run_feature_pipeline()
