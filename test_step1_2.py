"""
Script de Teste e Validação da Etapa 1.2.
Verifica a integridade dos atributos comportamentais de condução extraídos.
Execute no terminal com: python test_step1_2.py
"""

import sys
import os
from pathlib import Path

# Configura stdout para UTF-8 caso o terminal do Windows use cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import json

ROOT_DIR = Path(__file__).resolve().parent
PARQUET_FILE = ROOT_DIR / "data" / "processed" / "driving_behavior_features.parquet"
CSV_FILE = ROOT_DIR / "data" / "processed" / "driving_behavior_features.csv"
METADATA_FILE = ROOT_DIR / "data" / "processed" / "driving_behavior_metadata.json"


def test_step_1_2():
    print("=" * 68)
    print("INICIANDO TESTE DE VALIDACAO DA ETAPA 1.2 (ATRIBUTOS COMPORTAMENTAIS)")
    print("=" * 68)

    # 1. Verificar existência dos arquivos
    print("\n[1/4] Verificando se os arquivos processados existem no disco...")
    for file_path, name in [
        (PARQUET_FILE, "Parquet Processado"),
        (CSV_FILE, "CSV Processado"),
        (METADATA_FILE, "Metadados JSON"),
    ]:
        if not file_path.exists():
            print(f"[ERRO] Arquivo {name} nao encontrado em: {file_path}")
            sys.exit(1)
        size_kb = file_path.stat().st_size / 1024
        print(f"  [OK] {name}: Encontrado! ({size_kb:.1f} KB - {file_path.name})")

    # 2. Carregar dataset e validar dimensões
    print("\n[2/4] Carregando dataset comportamental com Pandas...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"  [OK] Total de trechos (janelas de 120s): {len(df):,d}")
    print(f"  [OK] Total de colunas: {len(df.columns)}")
    print(f"  [OK] Total de viagens cobertas: {df['Trip'].nunique():,d}")

    assert len(df) == 2531, f"Esperado 2.531 janelas, mas obteve {len(df)}"
    assert df.isna().sum().sum() == 0, "Existem valores nulos nos atributos calculados!"

    # 3. Validar consistência física dos atributos
    print("\n[3/4] Validando limites e consistencia fisica das variaveis...")
    assert (df["mean_speed_kmh"] >= 0).all(), "Velocidade media negativa encontrada!"
    assert (df["max_pos_accel_ms2"] <= 6.01).all(), "Aceleracao positiva acima do limite fisico!"
    assert (df["min_neg_accel_ms2"] >= -8.01).all(), "Desaceleracao abaixo do limite fisico!"
    assert (df["hard_braking_events"] >= 0).all(), "Contagem de frenagens negativa!"
    print("  [OK] Todos os atributos comportamentais estao dentro dos limites fisicos.")

    # 4. Comparativo de dois trechos reais contrastantes (Suave vs Agressivo)
    print("\n[4/4] Inspecionando dois trechos contrastantes reais da base:")
    
    # Trecho calmo: baixa taxa de frenagem e baixa variância de velocidade
    trecho_calmo = df[
        (df["hard_braking_rate_min"] <= 1.0) & 
        (df["std_speed_kmh"] < 10.0) & 
        (df["mean_speed_kmh"] > 25.0)
    ].iloc[0]

    # Trecho dinâmico/agressivo: alta taxa de frenagem brusca e alta oscilação
    trecho_agressivo = df[
        (df["hard_braking_rate_min"] >= 15.0) & 
        (df["rapid_accel_rate_min"] >= 15.0)
    ].iloc[0]

    print("\n  >>> AMOSTRA A: Conducao Suave/Constante (Candidato a Perfil Economico)")
    print(f"      • Viagem ID: {trecho_calmo['Trip']} | Janela: {trecho_calmo['window_id']}")
    print(f"      • Velocidade Media: {trecho_calmo['mean_speed_kmh']:.1f} km/h (Desvio Padrao: {trecho_calmo['std_speed_kmh']:.1f} km/h)")
    print(f"      • Aceleracao Maxima: {trecho_calmo['max_pos_accel_ms2']:.2f} m/s²")
    print(f"      • Frenagens Bruscas/min: {trecho_calmo['hard_braking_rate_min']:.2f}")
    print(f"      • Aceleracoes Bruscas/min: {trecho_calmo['rapid_accel_rate_min']:.2f}")
    print(f"      • Potencia Media: {trecho_calmo['mean_power_kw']:.2f} kW")

    print("\n  >>> AMOSTRA B: Conducao Agressiva/Oscilante (Candidato a Perfil Agressivo)")
    print(f"      • Viagem ID: {trecho_agressivo['Trip']} | Janela: {trecho_agressivo['window_id']}")
    print(f"      • Velocidade Media: {trecho_agressivo['mean_speed_kmh']:.1f} km/h (Desvio Padrao: {trecho_agressivo['std_speed_kmh']:.1f} km/h)")
    print(f"      • Aceleracao Maxima: {trecho_agressivo['max_pos_accel_ms2']:.2f} m/s²")
    print(f"      • Frenagens Bruscas/min: {trecho_agressivo['hard_braking_rate_min']:.2f}")
    print(f"      • Aceleracoes Bruscas/min: {trecho_agressivo['rapid_accel_rate_min']:.2f}")
    print(f"      • Potencia Media: {trecho_agressivo['mean_power_kw']:.2f} kW")

    print("\n" + "=" * 68)
    print("[SUCESSO] ETAPA 1.2 VALIDADA! Os atributos comportamentais")
    print("estao prontos para alimentar a clusterizacao (K-Means) na Etapa 1.3.")
    print("=" * 68)


if __name__ == "__main__":
    test_step_1_2()

