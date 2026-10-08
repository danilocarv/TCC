"""Aba 3 — Simulador em Tempo Real de Novos Condutores."""

import plotly.graph_objects as go
import streamlit as st

from app.config import (PROFILE_COLORS, PROFILE_ICONS, PROFILE_LABELS, PROFILE_DESCRIPTIONS,
                        SIMULATOR_PRESETS, STAR_COLOR)
from app.data_service import load_clusters, predict_profile
from app.tabs.tab1_overview import build_pca_figure

# (chave, rótulo, mín, máx, passo)
SLIDERS = [
    ("mean_speed_kmh", "Velocidade Média (km/h)", 0.0, 100.0, 0.5),
    ("std_speed_kmh", "Desvio Padrão da Velocidade (km/h)", 0.0, 30.0, 0.5),
    ("mean_pos_accel_ms2", "Aceleração Positiva Média (m/s²)", 0.0, 2.0, 0.01),
    ("max_pos_accel_ms2", "Aceleração Positiva Máxima (m/s²)", 0.5, 4.0, 0.05),
    ("hard_braking_rate_min", "Frenagens Bruscas por Minuto", 0.0, 5.0, 0.05),
    ("rapid_accel_rate_min", "Arrancadas Rápidas por Minuto", 0.0, 5.0, 0.05),
    ("jerk_rate_min", "Trancos Severos (Jerk) por Minuto", 0.0, 5.0, 0.05),
]


def _apply_preset(key: str):
    for k, v in SIMULATOR_PRESETS[key]["values"].items():
        st.session_state[f"sim_{k}"] = v


def render():
    st.subheader("🎮 Simulador em Tempo Real de Novos Condutores")
    if "sim_mean_speed_kmh" not in st.session_state:
        _apply_preset("urban")

    b = st.columns(3)
    for col, key in zip(b, ["eco", "urban", "sport"]):
        col.button(f"Carregar {SIMULATOR_PRESETS[key]['title']}", on_click=_apply_preset,
                   args=(key,), use_container_width=True)

    inputs, output = st.columns([1, 1.6])
    with inputs:
        st.markdown("#### Parâmetros da condução")
        for key, label, lo, hi, step in SLIDERS:
            st.slider(label, lo, hi, step=step, key=f"sim_{key}")
        st.caption("ℹ️ O K-Means usa as 6 primeiras variáveis. A taxa de *jerk* é exibida apenas "
                   "como indicador complementar, pois não integra o vetor de treino do modelo.")

    feats = {k: st.session_state[f"sim_{k}"] for k, *_ in SLIDERS}
    if feats["max_pos_accel_ms2"] < feats["mean_pos_accel_ms2"]:
        st.warning("A aceleração máxima está menor que a média — combinação fisicamente inconsistente.")
    cid, label, dists, pt = predict_profile(feats)

    with output:
        color = PROFILE_COLORS[label]
        st.markdown(
            f"<div class='profile-card' style='background:{color}'>"
            f"<small>Perfil predito pela IA</small><h3>{PROFILE_ICONS[label]} {label}</h3>"
            f"{PROFILE_DESCRIPTIONS[label]}</div>", unsafe_allow_html=True)

        fig = build_pca_figure(load_clusters(), opacity=0.25)
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]], mode="markers", name="⭐ Condutor simulado",
            marker=dict(symbol="star", size=30, color=STAR_COLOR, line=dict(color="black", width=2)),
            hovertemplate=f"Condutor simulado<br>{label}<br>PC1=%{{x:.2f}} · PC2=%{{y:.2f}}<extra></extra>",
        ))
        fig.update_layout(height=470)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Proximidade aos centróides (distância euclidiana no espaço normalizado)")
        names = [PROFILE_LABELS[i] for i in range(3)]
        dfig = go.Figure(go.Bar(
            x=dists, y=names, orientation="h",
            marker_color=[PROFILE_COLORS[n] for n in names],
            marker_line=dict(color=["black" if i == cid else "white" for i in range(3)], width=3),
            text=[f"{d:.2f}" for d in dists], textposition="outside",
        ))
        dfig.update_layout(height=220, template="plotly_white", xaxis_title="Distância (menor = mais próximo)",
                           yaxis=dict(autorange="reversed"), margin=dict(l=10, r=30, t=10, b=10))
        st.plotly_chart(dfig, use_container_width=True)
