"""
Configurações visuais e textuais do dashboard (cores, rótulos, presets, CSS).

Autor: Danilo Carvalho de Oliveira
Orientador: Prof. Me. Douglas Donizeti de Castilho Braz
"""

# Rótulos ordenados por agressividade (mesma ordem do DrivingProfileModel)
PROFILE_LABELS = {
    0: "Econômico / Suave",
    1: "Moderado / Regular",
    2: "Agressivo / Dinâmico",
}
PROFILE_ORDER = [PROFILE_LABELS[i] for i in range(3)]

PROFILE_COLORS = {
    "Econômico / Suave": "#2E9E5B",
    "Moderado / Regular": "#2F6FB5",
    "Agressivo / Dinâmico": "#D64541",
}
PROFILE_ICONS = {
    "Econômico / Suave": "🟢",
    "Moderado / Regular": "🔵",
    "Agressivo / Dinâmico": "🔴",
}
PROFILE_DESCRIPTIONS = {
    "Econômico / Suave": (
        "Condução estável, com baixa oscilação de velocidade, acelerações suaves e "
        "praticamente nenhuma frenagem brusca ou arrancada rápida. Perfil que favorece "
        "a eficiência energética e o aproveitamento da frenagem regenerativa."
    ),
    "Moderado / Regular": (
        "Comportamento equilibrado, típico do trânsito urbano/misto: acelerações "
        "moderadas e eventos bruscos ocasionais. Representa o padrão predominante da frota."
    ),
    "Agressivo / Dinâmico": (
        "Alta oscilação de velocidade, picos de aceleração elevados e frequência "
        "significativa de frenagens bruscas e arrancadas rápidas. Tende a elevar o "
        "consumo energético e o desgaste dos componentes."
    ),
}

FEATURE_NAMES = {
    "mean_speed_kmh": "Velocidade Média (km/h)",
    "std_speed_kmh": "Desvio Padrão da Velocidade (km/h)",
    "mean_pos_accel_ms2": "Aceleração Positiva Média (m/s²)",
    "max_pos_accel_ms2": "Aceleração Positiva Máxima (m/s²)",
    "hard_braking_rate_min": "Frenagens Bruscas / min",
    "rapid_accel_rate_min": "Arrancadas Rápidas / min",
}

STAR_COLOR = "#FFD400"
REGEN_COLOR = "#2E9E5B"
CONSUME_COLOR = "#E8743B"
SPEED_COLOR = "#2F6FB5"

# Presets do simulador (valores calibrados a partir dos centróides reais)
SIMULATOR_PRESETS = {
    "eco": {
        "title": "🟢 Exemplo: Motorista Eco em Rodovia",
        "values": {
            "mean_speed_kmh": 65.0, "std_speed_kmh": 5.0, "mean_pos_accel_ms2": 0.20,
            "max_pos_accel_ms2": 0.50, "hard_braking_rate_min": 0.0,
            "rapid_accel_rate_min": 0.0, "jerk_rate_min": 0.0,
        },
    },
    "urban": {
        "title": "🔵 Exemplo: Trânsito Misto Urbano",
        "values": {
            "mean_speed_kmh": 28.0, "std_speed_kmh": 16.0, "mean_pos_accel_ms2": 0.62,
            "max_pos_accel_ms2": 1.60, "hard_braking_rate_min": 0.30,
            "rapid_accel_rate_min": 0.05, "jerk_rate_min": 0.2,
        },
    },
    "sport": {
        "title": "🔴 Exemplo: Condutor Esportivo / Apressado",
        "values": {
            "mean_speed_kmh": 45.0, "std_speed_kmh": 24.0, "mean_pos_accel_ms2": 0.90,
            "max_pos_accel_ms2": 2.60, "hard_braking_rate_min": 1.80,
            "rapid_accel_rate_min": 1.50, "jerk_rate_min": 1.5,
        },
    },
}

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.tcc-header {
    background: linear-gradient(120deg, #0B3D2E 0%, #12664F 55%, #2F6FB5 100%);
    color: #fff; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
}
.tcc-header h1 {font-size: 1.55rem; margin: 0 0 .3rem 0; color: #fff;}
.tcc-header p {margin: .1rem 0; opacity: .92; font-size: .92rem;}
.badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    color: #fff; font-weight: 600; font-size: 1rem;
}
.profile-card {
    border-radius: 12px; padding: 1rem 1.2rem; color: #fff; margin-bottom: .6rem;
}
.profile-card h3 {margin: 0 0 .3rem 0; color: #fff;}
div[data-testid="stMetric"] {
    background: rgba(46,158,91,0.07); border: 1px solid rgba(46,158,91,0.25);
    border-radius: 10px; padding: .6rem .8rem;
}
</style>
"""
