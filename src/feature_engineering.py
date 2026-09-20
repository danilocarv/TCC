"""
Módulo de Engenharia de Atributos Comportamentais de Condução (Etapa 1.2).
Calcula variáveis cinemáticas ponto a ponto (aceleração longitudinal, jerk)
e extrai atributos comportamentais em janelas temporais de condução (120 segundos).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import os
import sys
import json
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import (
    RAW_EV_PARQUET,
    PROCESSED_BEHAVIOR_PARQUET,
    PROCESSED_BEHAVIOR_CSV,
    PROCESSED_BEHAVIOR_METADATA_JSON,
    WINDOW_SIZE_SECONDS,
    MIN_WINDOW_POINTS,
    HARD_BRAKING_THRESHOLD,
    RAPID_ACCEL_THRESHOLD,
    ensure_directories,
)


def compute_point_kinematics(df_telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variáveis cinemáticas ponto a ponto segundo a segundo:
    - Delta de tempo (dt em segundos);
    - Velocidade suavizada (para filtrar ruído de quantização dos sensores OBD);
    - Aceleração longitudinal (a em m/s²);
    - Taxa de variação da aceleração (jerk em m/s³);
    - Potência instantânea da bateria (kW).
    """
    print("-> [1/4] Calculando variáveis cinemáticas ponto a ponto (1 Hz)...")
    start_t = time.time()
    
    # Criar cópia para não alterar o DataFrame original
    df = df_telemetry.copy()
    
    # Garantir ordenação temporal estrita
    df.sort_values(by=["VehId", "Trip", "Timestamp(ms)"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Delta de tempo por viagem
    df["dt"] = df.groupby("Trip")["Timestamp(ms)"].diff() / 1000.0
    # Preencher primeiro ponto da viagem com 1.0s e limitar dt mínimo em 0.5s para evitar divisão por zero
    df["dt"] = df["dt"].fillna(1.0)
    # Se houver saltos anômalos de tempo (> 10s), limitar para 1.0s para não distorcer aceleração
    df.loc[df["dt"] <= 0, "dt"] = 1.0
    df.loc[df["dt"] > 10.0, "dt"] = 1.0
    
    # Conversão de velocidade: km/h para m/s
    df["v_ms"] = df["Vehicle Speed[km/h]"] / 3.6
    
    # Suavização móvel de 3 pontos para velocidade por viagem (remove ruído de degrau do sensor OBD)
    df["v_smooth_ms"] = (
        df.groupby("Trip")["v_ms"]
        .transform(lambda s: s.rolling(window=3, min_periods=1, center=True).mean())
    )
    
    # Aceleração longitudinal: a = dv / dt (m/s²)
    df["accel_ms2"] = (
        df.groupby("Trip")["v_smooth_ms"].diff() / df["dt"]
    ).fillna(0.0)
    
    # Limites físicos de segurança para veículos urbanos (Nissan Leaf: -8.0 a +6.0 m/s²)
    df["accel_ms2"] = df["accel_ms2"].clip(lower=-8.0, upper=6.0)
    
    # Jerk (variação de aceleração): j = da / dt (m/s³)
    df["jerk_ms3"] = (
        df.groupby("Trip")["accel_ms2"].diff() / df["dt"]
    ).fillna(0.0).clip(lower=-15.0, upper=15.0)
    
    # Potência instantânea da bateria em kW (P = V * I / 1000)
    # No VED: corrente negativa = descarga (consumo do motor elétrico); corrente positiva = regeneração
    # Padronizamos Potência Elétrica Consumida: P_consumida = - (Tensão * Corrente) / 1000
    # Assim, valores positivos indicam consumo de energia e valores negativos indicam recarga por regeneração
    df["power_kw"] = -(df["HV Battery Voltage[V]"] * df["HV Battery Current[A]"]) / 1000.0
    
    elapsed = time.time() - start_t
    print(f"   Cinemática calculada com sucesso em {elapsed:.1f}s!")
    return df


def extract_driving_behavior_windows(
    df_kinematics: pd.DataFrame,
    window_size_seconds: int = WINDOW_SIZE_SECONDS,
    min_points: int = MIN_WINDOW_POINTS,
) -> pd.DataFrame:
    """
    Segmenta as viagens em janelas operacionais de condução (120 segundos)
    e extrai os atributos comportamentais de cada trecho.
    """
    print(f"-> [2/4] Segmentando viagens em janelas temporais de {window_size_seconds}s...")
    start_t = time.time()
    
    # Tempo relativo decorrido dentro de cada viagem (em segundos)
    df = df_kinematics.copy()
    df["elapsed_s"] = df.groupby("Trip")["Timestamp(ms)"].transform(lambda x: (x - x.min()) / 1000.0)
    
    # Identificador da janela temporal dentro de cada viagem
    df["window_id"] = (df["elapsed_s"] // window_size_seconds).astype(int)
    
    feature_rows = []
    
    # Agrupamento por (VehId, Trip, window_id)
    grouped = df.groupby(["VehId", "Trip", "window_id"])
    total_groups = len(grouped)
    print(f"   Total de janelas brutas identificadas: {total_groups:,d}")
    
    for (veh_id, trip_id, win_id), group in grouped:
        # Filtrar janelas muito curtas (ex.: final de viagem com menos de min_points segundos)
        if len(group) < min_points:
            continue
        
        duration_s = float(group["dt"].sum())
        speeds = group["Vehicle Speed[km/h]"]
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
        
        # Eventos extremos comportamentais
        hard_brakes = int((accels <= HARD_BRAKING_THRESHOLD).sum())
        rapid_accels = int((accels >= RAPID_ACCEL_THRESHOLD).sum())
        high_jerks = int((jerks.abs() >= 3.0).sum())
        
        # Taxas normalizadas por minuto
        duration_min = max(duration_s / 60.0, 0.1)
        hard_brake_rate = float(hard_brakes / duration_min)
        rapid_accel_rate = float(rapid_accels / duration_min)
        
        # Proporção de tempo parado/ocioso (velocidade < 2.0 km/h)
        idle_points = (speeds < 2.0).sum()
        idle_ratio = float(idle_points / len(group))
        
        # Energia consumida aproximada no trecho: E (kWh) = integral(P * dt) / 3600
        # Potência (kW) * tempo (s) / 3600 s/h = kWh
        trecho_energy_kwh = float((powers * group["dt"]).sum() / 3600.0)
        
        feature_rows.append({
            "VehId": int(veh_id),
            "Trip": int(trip_id),
            "window_id": int(win_id),
            "duration_s": round(duration_s, 1),
            "sample_count": int(len(group)),
            # Atributos de Velocidade
            "mean_speed_kmh": round(float(speeds.mean()), 2),
            "max_speed_kmh": round(float(speeds.max()), 2),
            "std_speed_kmh": round(float(speeds.std()), 2) if len(speeds) > 1 else 0.0,
            # Atributos de Aceleração
            "mean_pos_accel_ms2": round(mean_pos_accel, 3),
            "max_pos_accel_ms2": round(max_pos_accel, 3),
            "mean_neg_accel_ms2": round(mean_neg_accel, 3),
            "min_neg_accel_ms2": round(min_neg_accel, 3),
            # Eventos Bruscos e Estabilidade
            "hard_braking_events": hard_brakes,
            "hard_braking_rate_min": round(hard_brake_rate, 2),
            "rapid_accel_events": rapid_accels,
            "rapid_accel_rate_min": round(rapid_accel_rate, 2),
            "high_jerk_events": high_jerks,
            "idle_ratio": round(idle_ratio, 3),
            # Contexto Energético do Trecho
            "mean_power_kw": round(float(powers.mean()), 2),
            "total_energy_kwh": round(trecho_energy_kwh, 4),
        })
    
    df_features = pd.DataFrame(feature_rows)
    # Tratar eventuais NaNs em desvios padrões caso ocorram
    df_features.fillna(0.0, inplace=True)
    
    elapsed = time.time() - start_t
    print(f"   Segmentação concluída em {elapsed:.1f}s!")
    print(f"   Total de janelas de condução válidas geradas: {len(df_features):,d}")
    return df_features


def save_features_and_metadata(df_features: pd.DataFrame):
    """Salva os atributos comportamentais extraídos em Parquet, CSV e JSON de metadados."""
    print(f"\n-> [3/4] Salvando dataset de atributos comportamentais...")
    ensure_directories()
    
    # 1. Salvar Parquet
    df_features.to_parquet(PROCESSED_BEHAVIOR_PARQUET, index=False, engine="pyarrow", compression="snappy")
    parquet_size_mb = os.path.getsize(PROCESSED_BEHAVIOR_PARQUET) / (1024 * 1024)
    print(f"   Arquivo Parquet gerado: {PROCESSED_BEHAVIOR_PARQUET.name} ({parquet_size_mb:.2f} MB)")
    
    # 2. Salvar CSV
    df_features.to_csv(PROCESSED_BEHAVIOR_CSV, index=False)
    csv_size_mb = os.path.getsize(PROCESSED_BEHAVIOR_CSV) / (1024 * 1024)
    print(f"   Arquivo CSV gerado: {PROCESSED_BEHAVIOR_CSV.name} ({csv_size_mb:.2f} MB)")
    
    # 3. Metadados e sumário estatístico
    feature_cols = [
        "mean_speed_kmh", "max_speed_kmh", "std_speed_kmh",
        "mean_pos_accel_ms2", "max_pos_accel_ms2", "mean_neg_accel_ms2", "min_neg_accel_ms2",
        "hard_braking_events", "hard_braking_rate_min",
        "rapid_accel_events", "rapid_accel_rate_min",
        "high_jerk_events", "idle_ratio"
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
        "description": "Atributos comportamentais de condução agregados em janelas de 120 segundos",
        "total_windows": int(len(df_features)),
        "total_trips_covered": int(df_features["Trip"].nunique()),
        "vehicles": [int(x) for x in sorted(df_features["VehId"].unique())],
        "feature_columns": feature_cols,
        "descriptive_statistics": stats_dict,
    }
    
    with open(PROCESSED_BEHAVIOR_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"   Metadados salvos em: {PROCESSED_BEHAVIOR_METADATA_JSON.name}")
    
    print("\n-> [4/4] ============ RESUMO DOS ATRIBUTOS COMPORTAMENTAIS ============")
    print(f"Total de Trechos de Condução (Janelas 120s): {len(df_features):,d}")
    print(f"Viagens Representadas: {df_features['Trip'].nunique():,d}")
    print("\nEstatísticas das Principais Variáveis do Condutor:")
    print(f"  • Velocidade Média:        {stats_dict['mean_speed_kmh']['mean']:.1f} km/h (Max: {stats_dict['mean_speed_kmh']['max']:.1f} km/h)")
    print(f"  • Desvio Padrão Velocidade:{stats_dict['std_speed_kmh']['mean']:.2f} km/h (Oscilação de velocidade)")
    print(f"  • Aceleração Média Pos.:   {stats_dict['mean_pos_accel_ms2']['mean']:.2f} m/s²")
    print(f"  • Aceleração Máxima:       {stats_dict['max_pos_accel_ms2']['mean']:.2f} m/s² (Pico máx: {stats_dict['max_pos_accel_ms2']['max']:.2f} m/s²)")
    print(f"  • Frenagens Bruscas/min:   {stats_dict['hard_braking_rate_min']['mean']:.2f} eventos/min (Máx: {stats_dict['hard_braking_rate_min']['max']:.2f})")
    print(f"  • Acelerações Bruscas/min: {stats_dict['rapid_accel_rate_min']['mean']:.2f} eventos/min (Máx: {stats_dict['rapid_accel_rate_min']['max']:.2f})")
    print("========================================================================\n")


def run_feature_pipeline():
    """Executa a pipeline completa da Etapa 1.2."""
    if not os.path.exists(RAW_EV_PARQUET):
        raise FileNotFoundError(f"Base de dados bruta não encontrada em: {RAW_EV_PARQUET}. Execute a Etapa 1.1 primeiro.")
    
    print("-> Carregando base de telemetria dos EVs...")
    df_telemetry = pd.read_parquet(RAW_EV_PARQUET)
    print(f"   Carregados {len(df_telemetry):,d} registros brutos.")
    
    df_kinematics = compute_point_kinematics(df_telemetry)
    df_features = extract_driving_behavior_windows(df_kinematics)
    save_features_and_metadata(df_features)


if __name__ == "__main__":
    run_feature_pipeline()

