"""
Script Interativo para Inspecionar Viagens Reais dos Veículos Elétricos.
Permite visualizar qualquer uma das 491 viagens gravadas pelos sensores do carro.

Como usar:
1) Modo interativo (digitar na tela):
   python inspect_trip.py

2) Passando o ID direto na linha de comando:
   python inspect_trip.py 1582
   python inspect_trip.py 1561
   python inspect_trip.py 1568
"""

import sys
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
FEATURES_FILE = ROOT_DIR / "data" / "processed" / "driving_behavior_features.parquet"
TELEMETRY_FILE = ROOT_DIR / "data" / "raw" / "ev_telemetry.parquet"


def print_trip_details(trip_id: int, df_features: pd.DataFrame, veh_id: int = None):
    """Exibe o relatório detalhado de uma viagem específica (e veículo, se especificado)."""
    if veh_id is not None:
        trip_features = df_features[(df_features["Trip"] == trip_id) & (df_features["VehId"] == veh_id)]
    else:
        trip_features = df_features[df_features["Trip"] == trip_id]

    if trip_features.empty:
        print(f"\n[AVISO] Viagem {trip_id} nao encontrada na base processada!")
        print("Dica: Exemplos de viagens existentes: 1558, 1561, 1568, 1572, 1582, 1601, 1674, 1707, etc.")
        return

    # Se houver mais de um veículo com este mesmo Trip ID e veh_id não foi especificado, detalhar cada um
    unique_vehs = trip_features["VehId"].unique().tolist()
    if len(unique_vehs) > 1 and veh_id is None:
        print(f"\n[INFO] O Trip ID {trip_id} foi registrado para {len(unique_vehs)} veiculos diferentes: {unique_vehs}")
        for v in unique_vehs:
            print_trip_details(trip_id, df_features, veh_id=v)
        return

    veh_id = int(trip_features["VehId"].iloc[0])
    total_windows = len(trip_features)
    duration_s = trip_features["duration_s"].sum()
    duration_min = duration_s / 60.0

    # Médias agregadas da viagem
    mean_speed = trip_features["mean_speed_kmh"].mean()
    max_speed = trip_features["max_speed_kmh"].max()
    speed_std = trip_features["std_speed_kmh"].mean()
    max_accel = trip_features["max_pos_accel_ms2"].max()
    min_accel = trip_features["min_neg_accel_ms2"].min()
    total_hard_brakes = trip_features["hard_braking_events"].sum()
    total_rapid_accels = trip_features["rapid_accel_events"].sum()
    mean_idle = trip_features["idle_ratio"].mean() * 100
    mean_power = trip_features["mean_power_kw"].mean()
    total_energy = trip_features["total_energy_kwh"].sum()

    print("\n" + "=" * 70)
    print(f"🚗 RELATORIO DA VIAGEM REAL - TRIP ID: {trip_id}")
    print("=" * 70)
    print(f"* Veiculo (ID / Modelo):      Veiculo {veh_id} (Nissan Leaf 2013 - 100% Eletrico)")
    print(f"* Duracao Total da Viagem:    {duration_s:.0f} segundos (~{duration_min:.1f} minutos)")
    print(f"* Trechos de 120s analisados: {total_windows} trecho(s)")
    print("-" * 70)
    print("📊 METRICAS COMPORTAMENTAIS DO CONDUTOR:")
    print(f"  - Velocidade Media:         {mean_speed:.1f} km/h")
    print(f"  - Velocidade Maxima:        {max_speed:.1f} km/h")
    print(f"  - Variacao de Velocidade:   {speed_std:.1f} km/h (desvio padrao medio)")
    print(f"  - Aceleracao Maxima:        {max_accel:.2f} m/s² (pico de arrancada)")
    print(f"  - Frenagem Mais Forte:      {min_accel:.2f} m/s² (pico de desaceleracao)")
    print(f"  - Freadas Bruscas no Total: {total_hard_brakes} evento(s) (aceleracao <= -2.0 m/s²)")
    print(f"  - Arrancadas Fortes Total:  {total_rapid_accels} evento(s) (aceleracao >= 2.0 m/s²)")
    print(f"  - Tempo Parado/Semaforos:   {mean_idle:.1f}% do tempo total")
    print("-" * 70)
    print("⚡ METRICAS DE CONSUMO ELETRICO:")
    print(f"  - Potencia Media Solicitada:{mean_power:.2f} kW")
    print(f"  - Energia Total Consumida:  {total_energy:.3f} kWh")
    print("-" * 70)

    # Detalhamento trecho a trecho se a viagem tiver mais de 1 janela
    if total_windows > 1:
        print("📋 EVOLUCAO POR TRECHO DE 2 MINUTOS (120s):")
        print("Trecho | Vel.Media | Vel.Max | Freadas | Arrancadas | Potencia(kW) | Energia(kWh)")
        for _, row in trip_features.iterrows():
            print(
                f"  {int(row['window_id']):02d}   |"
                f"  {row['mean_speed_kmh']:>5.1f}   |"
                f"  {row['max_speed_kmh']:>5.1f}  |"
                f"   {int(row['hard_braking_events']):>2d}    |"
                f"     {int(row['rapid_accel_events']):>2d}     |"
                f"    {row['mean_power_kw']:>6.2f}    |"
                f"   {row['total_energy_kwh']:>7.3f}"
            )
        print("-" * 70)

    # Taxas comportamentais por minuto (normalizadas pela duração da viagem)
    effective_min = max(duration_min, 0.5)
    hard_brakes_per_min = total_hard_brakes / effective_min
    rapid_accels_per_min = total_rapid_accels / effective_min

    # Diagnóstico preliminar de estilo baseado na distribuição estatística real
    if hard_brakes_per_min <= 0.20 and max_accel < 1.8:
        estilo = "🟢 SUAVE / ECONOMICO (Condutor calmo, quase sem freadas bruscas, velocidade uniforme)"
    elif hard_brakes_per_min >= 0.80 or rapid_accels_per_min >= 0.60 or speed_std >= 20.0:
        estilo = "🔴 AGRESSIVO / DINAMICO (Freadas bruscas frequentes, arrancadas fortes, alta oscilacao)"
    else:
        estilo = "🟡 MODERADO / REGULAR (Comportamento equilibrado no transito urbano/rodoviario)"

    print(f"🎯 Diagnostico Preliminar do Estilo: {estilo}")
    print(f"   (Freadas/min: {hard_brakes_per_min:.2f} | Arrancadas/min: {rapid_accels_per_min:.2f})")
    print("=" * 70 + "\n")


def main():
    if not FEATURES_FILE.exists():
        print(f"[ERRO] Base de dados nao encontrada em: {FEATURES_FILE}")
        print("Execute a Etapa 1.2 primeiro com: python src/feature_engineering.py")
        sys.exit(1)

    df_features = pd.read_parquet(FEATURES_FILE)
    available_trips = sorted(df_features["Trip"].unique().tolist())

    # Se o usuário passou argumentos na linha de comando
    if len(sys.argv) > 1:
        # Se passou apenas um argumento (ex: python inspect_trip.py 1582)
        # ou se passou múltiplos argumentos, pega o último número inteiro digitado
        integers_passed = []
        for arg in sys.argv[1:]:
            if arg.isdigit():
                integers_passed.append(int(arg))
        
        if integers_passed:
            # Se passou apenas 1 número, é o Trip ID
            # Se passou 2 números, o primeiro é Trip ID e o segundo é VehId
            if len(integers_passed) == 1:
                trip_id = integers_passed[0]
                print_trip_details(trip_id, df_features)
            else:
                trip_id = integers_passed[0]
                veh_id = integers_passed[1]
                print_trip_details(trip_id, df_features, veh_id=veh_id)
            return
        else:
            print(f"[ERRO] Nenhum ID valido encontrado nos argumentos: {sys.argv[1:]}")
            sys.exit(1)

    # Modo Interativo no Terminal
    print("=" * 70)
    print("🔍 INSPECIONADOR INTERATIVO DE VIAGENS DOS VEICULOS ELETRICOS")
    print("=" * 70)
    print(f"Total de viagens reais disponiveis no dataset: {len(available_trips)}")
    print("Sugestoes para testar os 3 estilos:")
    print("  * Suaves/Economicas: 1582, 1602, 1727, 1781, 1816")
    print("  * Moderadas/Normais: 1568, 1578, 1585, 1599, 1674")
    print("  * Agressivas/Dinamicas: 1561, 1601, 1625, 1707, 1749")
    print("-" * 70)

    while True:
        try:
            escolha = input(
                "Digite o numero da Trip ID (ou 'r' para aleatoria, 'q' para sair): "
            ).strip()
            if escolha.lower() == "q":
                print("Encerrando inspecionador.")
                break
            elif escolha.lower() == "r":
                trip_id = int(np.random.choice(available_trips))
                print(f"\n-> Sorteada aleatoriamente a viagem Trip ID: {trip_id}")
                print_trip_details(trip_id, df_features)
            elif escolha.isdigit():
                trip_id = int(escolha)
                print(f"\n-> Consultando viagem solicitada Trip ID: {trip_id}...")
                print_trip_details(trip_id, df_features)
            else:
                print("Entrada invalida. Digite o numero da viagem (ex: 1582), 'r' ou 'q'.")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando.")
            break


if __name__ == "__main__":
    main()
