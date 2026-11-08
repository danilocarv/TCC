"""
Generates high-resolution infographic poster: 28_dicionario_variaveis_ev_trips.png
Visual dictionary for all 47 attributes in ev_trips_complete_dataset.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_FIGS_DIR = Path("outputs/figures/trips")
ARTIFACT_DIR = Path("C:/Users/danil/.gemini/antigravity/brain/32a12848-7a2c-439d-92f4-f6e0db509cff")

OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Infographic setup
fig = plt.figure(figsize=(26, 18), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# Background
ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="#F8FAFC", edgecolor="none"))

# Header banner
ax.add_patch(patches.Rectangle((0.02, 0.91), 0.96, 0.075, facecolor="#0F172A", edgecolor="none"))
ax.text(0.04, 0.955, "DICIONÁRIO DE DADOS DE VIAGENS COMPLETAS: ev_trips_complete_dataset", 
        fontsize=22, fontweight="bold", color="#F8FAFC", va="center")
ax.text(0.04, 0.925, "Base VED (Vehicle Energy Dataset) | 504 Viagens (486 Operacionais Válidas) | Nissan Leaf 2013 (24 kWh) | 47 Atributos Físicos Consolidados", 
        fontsize=12, color="#94A3B8", va="center")
ax.add_patch(patches.Rectangle((0.84, 0.925), 0.12, 0.045, facecolor="#2563EB", edgecolor="none"))
ax.text(0.90, 0.947, "47 VARIÁVEIS", fontsize=11, fontweight="bold", color="white", ha="center", va="center")

# Layout specifications: 3 columns of cards
# Col 1: Metadados (7) + Clima HVAC (5)
# Col 2: Distância & Cinemática (12) + Perfis (5)
# Col 3: Bateria (6) + Energia & Regeneração (5) + Altimetria (7)

def draw_card(ax, x, y, w, h, title, subtitle, color_primary, color_bg, items):
    # Card background and border
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="white", edgecolor=color_primary, linewidth=1.8
    ))
    # Header box
    ax.add_patch(patches.FancyBboxPatch(
        (x, y + h - 0.048), w, 0.048,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=color_primary, edgecolor="none"
    ))
    ax.text(x + 0.015, y + h - 0.022, title, fontsize=12.5, fontweight="bold", color="white", va="center")
    ax.text(x + w - 0.015, y + h - 0.022, subtitle, fontsize=10, fontweight="bold", color="#E2E8F0", ha="right", va="center")

    # Items
    y_cursor = y + h - 0.065
    step = (h - 0.060) / max(len(items), 1)

    for item in items:
        var_name, var_type, desc, is_target = item
        
        # Target highlight
        if is_target:
            ax.add_patch(patches.Rectangle((x + 0.008, y_cursor - 0.012), w - 0.016, 0.038, facecolor="#FEF08A", edgecolor="#EAB308", linewidth=1.2))
            badge = " [ALVO DA REGRESSÃO]"
            name_color = "#854D0E"
        else:
            badge = ""
            name_color = "#0F172A"

        # Variable line
        ax.text(x + 0.015, y_cursor + 0.010, f"{var_name}{badge}", fontsize=10.5, fontweight="bold", color=name_color, family="monospace")
        ax.text(x + 0.015 + len(var_name)*0.006 + (0.11 if is_target else 0.0), y_cursor + 0.010, f"({var_type})", fontsize=8.5, color="#64748B", style="italic")
        ax.text(x + 0.015, y_cursor - 0.008, desc, fontsize=9.2, color="#334155")
        
        y_cursor -= step

# Data definitions
col1_meta = [
    ("trip_id", "string", "Identificador único da viagem no formato '{VehId}_{Trip}' (ex: '10_1558')", False),
    ("VehId", "int", "Código identificador do veículo elétrico (10, 455 ou 541 — Nissan Leaf 2013)", False),
    ("Trip", "int", "Número sequencial da viagem realizada pelo veículo", False),
    ("DayNum", "float", "Dia do ano em que a viagem ocorreu (permite analisar sazonalidade e calendário)", False),
    ("duration_s", "segundos", "Duração total da viagem medida pelo somatório temporal dos passos dt", False),
    ("sample_count", "inteiro", "Quantidade total de registros de telemetria a 1 Hz coletados na viagem", False),
    ("is_valid_trip", "booleano", "Flag de consistência operacional: True se duração >= 60s e distância >= 0.5km", False)
]

col1_hvac = [
    ("aux_hvac_energy_kwh", "kWh", "Energia elétrica consumida exclusivamente pelo aquecedor PTC e ar-condicionado", False),
    ("hvac_share_percent", "%", "Fração percentual da energia bruta que foi drenada pelo sistema térmico de cabine", False),
    ("mean_oat_c", "°C", "Temperatura ambiente externa média durante o trajeto (média de OAT[DegC])", False),
    ("min_oat_c", "°C", "Menor temperatura externa observada na viagem (atinge até -14.45 °C no inverno)", False),
    ("max_oat_c", "°C", "Maior temperatura externa observada na viagem (atinge até +35.51 °C no verão)", False)
]

col2_kinematics = [
    ("distance_km", "km", "Distância real percorrida integrada do sensor de velocidade de roda (∑ v·dt/3600)", False),
    ("distance_gps_km", "km", "Distância acumulada via coordenadas de GPS pela fórmula de Haversine", False),
    ("mean_speed_kmh", "km/h", "Velocidade média desenvolvida pelo veículo durante a viagem", False),
    ("max_speed_kmh", "km/h", "Pico máximo de velocidade atingido durante o percurso", False),
    ("std_speed_kmh", "km/h", "Desvio padrão da velocidade (indica tráfego para-e-anda e fluidez)", False),
    ("idle_ratio", "0 a 1", "Fração do tempo da viagem em que o veículo esteve parado (velocidade < 1 km/h)", False),
    ("mean_pos_accel_ms2", "m/s²", "Aceleração positiva média durante momentos de tração (a > 0.1 m/s²)", False),
    ("max_pos_accel_ms2", "m/s²", "Arrancada mais forte registrada na viagem (pico de aceleração positiva)", False),
    ("mean_neg_accel_ms2", "m/s²", "Desaceleração média observada durante as frenagens (a < -0.1 m/s²)", False),
    ("min_neg_accel_ms2", "m/s²", "Frenagem mais brusca registrada na viagem (pico negativo de desaceleração)", False),
    ("hard_braking_events", "contagem", "Total de frenagens bruscas críticas na viagem (desaceleração <= -2.5 m/s²)", False),
    ("rapid_accel_events", "contagem", "Total de arrancadas intensas na viagem (aceleração >= +2.0 m/s²)", False)
]

col2_profiles = [
    ("driver_profile_dominant", "string", "Rótulo do perfil predominante ('Econômico', 'Moderado' ou 'Agressivo')", False),
    ("driver_profile_dominant_id", "int (0,1,2)", "Código numérico do perfil (0: Econômico / 1: Moderado / 2: Agressivo)", False),
    ("eco_window_ratio", "0 a 1", "Proporção do tempo da viagem conduzida no modo suave/econômico", False),
    ("moderate_window_ratio", "0 a 1", "Proporção do tempo da viagem conduzida no modo moderado/regular", False),
    ("aggressive_window_ratio", "0 a 1", "Proporção do tempo da viagem conduzida no modo agressivo/dinâmico", False)
]

col3_battery = [
    ("initial_soc_percent", "%", "Estado de carga da bateria (SOC) registrado no primeiro segundo da viagem", False),
    ("final_soc_percent", "%", "Estado de carga da bateria (SOC) registrado no último segundo da viagem", False),
    ("delta_soc_percent", "%", "Queda percentual do estado de carga durante a viagem (initial_soc - final_soc)", False),
    ("initial_voltage_v", "Volts", "Tensão elétrica do pacote de baterias antes do início do deslocamento", False),
    ("final_voltage_v", "Volts", "Tensão elétrica do pacote de baterias no destino final da viagem", False),
    ("mean_voltage_v", "Volts", "Tensão elétrica média da bateria observada durante todo o percurso", False)
]

col3_energy = [
    ("energy_gross_kwh", "kWh", "Consumo bruto de energia da tração e sistemas nos momentos de descarga (P > 0)", False),
    ("energy_regen_kwh", "kWh", "Energia recuperada pela frenagem regenerativa e recarregada na bateria (P < 0)", False),
    ("energy_net_kwh", "kWh", "Consumo Real Líquido da Bateria: energy_gross - energy_regen (Alvo do XGBoost)", True),
    ("regen_efficiency_ratio", "0 a 1", "Taxa de eficiência de regeneração da viagem (energy_regen / energy_gross)", False),
    ("net_consumption_kwh_100km", "kWh/100km", "Consumo líquido normalizado para o padrão automotivo (kWh por 100 km rodados)", False)
]

col3_elevation = [
    ("elevation_gain_m", "metros", "Ganho acumulado de subida na rota com filtro deadband de 0.5m no DEM SRTM 30m", False),
    ("elevation_loss_m", "metros", "Perda acumulada de altitude (descidas) com filtro deadband de 0.5m", False),
    ("net_elevation_m", "metros", "Desnível líquido final da viagem entre destino e origem (h_final - h_inicial)", False),
    ("initial_elevation_m", "metros", "Altitude do ponto de partida da viagem em relação ao nível do mar", False),
    ("final_elevation_m", "metros", "Altitude do ponto de chegada da viagem em relação ao nível do mar", False),
    ("climbing_grade_percent", "%", "Inclinação média ponderada de subida da rota (elevation_gain / distancia * 100)", False),
    ("net_grade_percent", "%", "Gradiente líquido de rampa da viagem (net_elevation / distancia * 100)", False)
]

# Draw Column 1 (x: 0.02 -> 0.32)
draw_card(ax, 0.025, 0.495, 0.300, 0.395, "1. METADADOS E IDENTIFICAÇÃO", "7 Variáveis", "#334155", "#F1F5F9", col1_meta)
draw_card(ax, 0.025, 0.070, 0.300, 0.405, "5. CLIMA E CARGAS TÉRMICAS (HVAC)", "5 Variáveis", "#EA580C", "#FFF7ED", col1_hvac)

# Draw Column 2 (x: 0.35 -> 0.65)
draw_card(ax, 0.350, 0.340, 0.300, 0.550, "2. DISTÂNCIA E CINEMÁTICA VEICULAR", "12 Variáveis", "#0284C7", "#F0F9FF", col2_kinematics)
draw_card(ax, 0.350, 0.070, 0.300, 0.250, "7. PERFIL DE CONDUÇÃO (FASE 1)", "5 Variáveis", "#DC2626", "#FEF2F2", col2_profiles)

# Draw Column 3 (x: 0.675 -> 0.975)
draw_card(ax, 0.675, 0.615, 0.300, 0.275, "3. BATERIA E ESTADO ELÉTRICO", "6 Variáveis", "#7C3AED", "#F5F3FF", col3_battery)
draw_card(ax, 0.675, 0.340, 0.300, 0.255, "4. ENERGIA E REGENERAÇÃO", "5 Variáveis", "#059669", "#ECFDF5", col3_energy)
draw_card(ax, 0.675, 0.070, 0.300, 0.250, "6. ALTIMETRIA E TOPOGRAFIA (SRTM 30m)", "7 Variáveis", "#16A34A", "#F0FDF4", col3_elevation)

# Footer bar
ax.add_patch(patches.Rectangle((0.02, 0.015), 0.96, 0.038, facecolor="#E2E8F0", edgecolor="none"))
footer_text = "Fonte de Dados: Vehicle Energy Dataset (VED) | Modelo de Relevo: NASA/ESA SRTM 30m DEM (N42W084.hgt) | Módulo Gerador: src/trip_feature_extraction.py"
ax.text(0.04, 0.034, footer_text, fontsize=10, color="#475569", va="center")
ax.text(0.96, 0.034, "TCC - Predição de Consumo Energético em Veículos Elétricos", fontsize=10, fontweight="bold", color="#1E293B", ha="right", va="center")

save_path = OUTPUT_FIGS_DIR / "28_dicionario_variaveis_ev_trips.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Infográfico salvo em: {save_path}")

# Copy to artifact dir
if ARTIFACT_DIR.exists():
    dst = ARTIFACT_DIR / "28_dicionario_variaveis_ev_trips.png"
    shutil.copy2(save_path, dst)
    print(f"Copiado para o diretório de artefatos: {dst}")
