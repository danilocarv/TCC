"""
Script de Teste e Validação da Etapa 1.1.
Verifica a integridade dos dados extraídos dos veículos 100% elétricos.
Execute no terminal com: python test_step1.py
"""

import sys
import os
from pathlib import Path

# Configura stdout para UTF-8 caso o terminal do Windows use cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import json

# Caminhos dos arquivos gerados
ROOT_DIR = Path(__file__).resolve().parent
PARQUET_FILE = ROOT_DIR / "data" / "raw" / "ev_telemetry.parquet"
CSV_FILE = ROOT_DIR / "data" / "raw" / "ev_telemetry.csv"
METADATA_FILE = ROOT_DIR / "data" / "raw" / "ev_dataset_metadata.json"


def test_step_1():
    print("=" * 65)
    print("INICIANDO TESTE DE VALIDACAO DA ETAPA 1.1 (DADOS DOS EVs)")
    print("=" * 65)

    # 1. Verificar existência dos arquivos
    print("\n[1/4] Verificando se os arquivos foram criados no disco...")
    for file_path, name in [
        (PARQUET_FILE, "Parquet"),
        (CSV_FILE, "CSV"),
        (METADATA_FILE, "Metadados JSON"),
    ]:
        if not file_path.exists():
            print(f"[ERRO] O arquivo {name} nao foi encontrado em: {file_path}")
            sys.exit(1)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {name}: Encontrado! Tamanho: {size_mb:.2f} MB ({file_path.name})")

    # 2. Ler o arquivo Parquet e verificar dimensões
    print("\n[2/4] Carregando a base compactada Parquet com Pandas...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"  [OK] Total de linhas carregadas: {len(df):,d}")
    print(f"  [OK] Total de colunas: {len(df.columns)}")
    print(f"  [OK] Colunas presentes: {list(df.columns)}")

    assert len(df) == 476308, f"Esperado 476.308 linhas, mas obteve {len(df)}"
    assert len(df.columns) == 13, f"Esperado 13 colunas, mas obteve {len(df.columns)}"

    # 3. Testar veículos e valores ausentes (Nulos)
    print("\n[3/4] Validando veiculos e consistencia de dados...")
    veh_ids = sorted(df["VehId"].unique().tolist())
    print(f"  [OK] IDs de Veiculos encontrados: {veh_ids} (Apenas os EVs puros: 10, 455 e 541)")
    assert veh_ids == [10, 455, 541], "Erro nos IDs dos veiculos eletricos"

    nulos_total = df.isna().sum().sum()
    print(f"  [OK] Total de valores nulos nas variaveis operacionais: {nulos_total}")
    assert nulos_total == 0, "Existem valores nulos inesperados na base!"

    # 4. Amostra de uma viagem real
    trip_exemplo = df["Trip"].iloc[0]
    df_trip = df[df["Trip"] == trip_exemplo]
    duracao_segundos = len(df_trip)
    vel_media = df_trip["Vehicle Speed[km/h]"].mean()
    vel_max = df_trip["Vehicle Speed[km/h]"].max()
    tensao_media = df_trip["HV Battery Voltage[V]"].mean()

    print("\n[4/4] Inspecionando dados de uma viagem real:")
    print(f"  * Viagem Exemplo (Trip ID): {trip_exemplo} (Veiculo {df_trip['VehId'].iloc[0]})")
    print(f"  * Duracao: {duracao_segundos} segundos (~{duracao_segundos/60:.1f} minutos)")
    print(f"  * Velocidade Media: {vel_media:.1f} km/h | Maxima: {vel_max:.1f} km/h")
    print(f"  * Tensao Media da Bateria: {tensao_media:.1f} V")
    print(
        f"  * Variacao da Carga (SOC): de {df_trip['HV Battery SOC[%]'].iloc[0]:.1f}% a {df_trip['HV Battery SOC[%]'].iloc[-1]:.1f}%"
    )

    print("\n" + "=" * 65)
    print("[SUCESSO] TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("A base dos veiculos eletricos esta 100% integra e pronta.")
    print("=" * 65)


if __name__ == "__main__":
    test_step_1()

