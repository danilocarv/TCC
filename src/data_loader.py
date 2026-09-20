"""
Módulo de Ingestão e Extração de Dados dos Veículos Elétricos (VED).
Filtra e consolida exclusivamente os dados dos veículos 100% elétricos (Nissan Leaf 2013).
"""

import os
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import glob
import time
import json
import pandas as pd
import numpy as np

from src.config import (
    ensure_directories,
    VED_DATA_DIR,
    VED_STATIC_PHEV_EV_FILE,
    EV_VEHICLE_IDS,
    COLUMNS_DYNAMIC,
    DTYPES_DYNAMIC,
    RAW_EV_PARQUET,
    RAW_EV_CSV,
    RAW_EV_METADATA_JSON,
)


def load_static_metadata() -> pd.DataFrame:
    """Carrega os metadados estáticos dos veículos elétricos (Nissan Leaf 2013)."""
    print("-> Lendo metadados estáticos dos veículos elétricos...")
    if not os.path.exists(VED_STATIC_PHEV_EV_FILE):
        raise FileNotFoundError(f"Arquivo estático não encontrado: {VED_STATIC_PHEV_EV_FILE}")
    
    df_static = pd.read_excel(VED_STATIC_PHEV_EV_FILE)
    ev_static = df_static[df_static["VehId"].isin(EV_VEHICLE_IDS)].copy()
    print(f"   Veículos elétricos encontrados nos metadados: {len(ev_static)}")
    return ev_static


def extract_pure_ev_dynamic_data() -> pd.DataFrame:
    """
    Varre todos os 54 arquivos dinâmicos semanais do VED, filtra
    exclusivamente os registros dos veículos elétricos e consolida
    em um único DataFrame ordenado cronologicamente por viagem.
    """
    ensure_directories()
    
    # Localizar todos os arquivos semanais CSV (Part 1 e Part 2)
    csv_pattern = os.path.join(VED_DATA_DIR, "VED_DynamicData_Part*", "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV dinâmico encontrado em: {csv_pattern}")
    
    print(f"-> Localizados {len(csv_files)} arquivos semanais do VED para processamento.")
    start_time = time.time()
    
    ev_chunks = []
    total_files = len(csv_files)
    
    for idx, file_path in enumerate(csv_files, start=1):
        filename = os.path.basename(file_path)
        try:
            # Lemos apenas as colunas relevantes para economizar memória e acelerar o I/O
            chunk = pd.read_csv(file_path, usecols=COLUMNS_DYNAMIC, low_memory=False)
            ev_subset = chunk[chunk["VehId"].isin(EV_VEHICLE_IDS)].copy()
            
            if not ev_subset.empty:
                # Aplicar conversão de tipos otimizada
                for col, dtype in DTYPES_DYNAMIC.items():
                    if col in ev_subset.columns:
                        ev_subset[col] = pd.to_numeric(ev_subset[col], errors="coerce").astype(dtype)
                
                ev_chunks.append(ev_subset)
                print(f"   [{idx:02d}/{total_files:02d}] {filename} -> {len(ev_subset):>6,d} linhas EV")
            else:
                print(f"   [{idx:02d}/{total_files:02d}] {filename} ->      0 linhas EV")
        except Exception as err:
            print(f"   [AVISO] Erro ao processar {filename}: {err}")
    
    if not ev_chunks:
        raise ValueError("Nenhum registro de veículo elétrico foi encontrado nos arquivos!")
    
    print("\n-> Consolidando e ordenando os dados...")
    df_ev = pd.concat(ev_chunks, ignore_index=True)
    
    # Ordenar rigorosamente por veículo, viagem e timestamp temporal
    df_ev.sort_values(by=["VehId", "Trip", "Timestamp(ms)"], inplace=True)
    df_ev.reset_index(drop=True, inplace=True)
    
    elapsed = time.time() - start_time
    print(f"-> Extração concluída com sucesso em {elapsed:.1f} segundos!")
    print(f"   Total de registros EV consolidados: {len(df_ev):,d}")
    print(f"   Total de viagens distintas (Trip): {df_ev['Trip'].nunique():,d}")
    
    return df_ev


def save_and_summarize(df_ev: pd.DataFrame, df_static: pd.DataFrame):
    """Salva os dados consolidados em Parquet e CSV e gera metadados em JSON."""
    print(f"\n-> Salvando dados consolidados em formato Parquet: {RAW_EV_PARQUET}...")
    df_ev.to_parquet(RAW_EV_PARQUET, index=False, engine="pyarrow", compression="snappy")
    parquet_size_mb = os.path.getsize(RAW_EV_PARQUET) / (1024 * 1024)
    print(f"   Arquivo Parquet gerado com sucesso ({parquet_size_mb:.2f} MB).")
    
    print(f"-> Salvando cópia em CSV para conferência direta: {RAW_EV_CSV}...")
    df_ev.to_csv(RAW_EV_CSV, index=False)
    csv_size_mb = os.path.getsize(RAW_EV_CSV) / (1024 * 1024)
    print(f"   Arquivo CSV gerado com sucesso ({csv_size_mb:.2f} MB).")
    
    # Estatísticas por veículo
    veh_stats = {}
    for veh_id in sorted(df_ev["VehId"].unique()):
        veh_df = df_ev[df_ev["VehId"] == veh_id]
        veh_stats[int(veh_id)] = {
            "total_records": int(len(veh_df)),
            "total_trips": int(veh_df["Trip"].nunique()),
            "speed_kmh_mean": float(round(veh_df["Vehicle Speed[km/h]"].mean(), 2)),
            "speed_kmh_max": float(round(veh_df["Vehicle Speed[km/h]"].max(), 2)),
            "current_a_min": float(round(veh_df["HV Battery Current[A]"].min(), 2)),
            "current_a_max": float(round(veh_df["HV Battery Current[A]"].max(), 2)),
            "voltage_v_mean": float(round(veh_df["HV Battery Voltage[V]"].mean(), 2)),
            "soc_pct_mean": float(round(veh_df["HV Battery SOC[%]"].mean(), 2)),
        }
    
    # Valores nulos por coluna
    null_counts = {col: int(df_ev[col].isna().sum()) for col in df_ev.columns}
    
    metadata = {
        "dataset_name": "Vehicle Energy Dataset (VED) - Pure Electric Vehicles (EV)",
        "models": "Nissan Leaf 2013 (24 kWh battery)",
        "vehicle_ids": [int(x) for x in sorted(df_ev["VehId"].unique())],
        "total_rows": int(len(df_ev)),
        "total_trips": int(df_ev["Trip"].nunique()),
        "columns": list(df_ev.columns),
        "null_counts": null_counts,
        "per_vehicle_summary": veh_stats,
    }
    
    with open(RAW_EV_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"   Metadados salvos em JSON: {RAW_EV_METADATA_JSON}")
    
    print("\n================ RESUMO DOS DADOS DOS VEÍCULOS ELÉTRICOS ================")
    print(f"Total de Registros: {metadata['total_rows']:,d}")
    print(f"Total de Viagens Reais: {metadata['total_trips']:,d}")
    print("\nDetalhamento por Veículo:")
    for v_id, s in veh_stats.items():
        print(f"  • Veículo ID {v_id}: {s['total_records']:,d} linhas | {s['total_trips']} viagens | Vel. Média: {s['speed_kmh_mean']} km/h | Vel. Máx: {s['speed_kmh_max']} km/h")
    print("========================================================================\n")


def run_pipeline():
    """Executa a pipeline completa da Etapa 1.1."""
    df_static = load_static_metadata()
    df_ev = extract_pure_ev_dynamic_data()
    save_and_summarize(df_ev, df_static)


if __name__ == "__main__":
    run_pipeline()
