"""Aba 2 — Inspetor Interativo de Viagens Reais (VED)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.config import (PROFILE_COLORS, PROFILE_ICONS, PROFILE_ORDER,
                        REGEN_COLOR, CONSUME_COLOR, SPEED_COLOR)
from app.data_service import load_clusters, trip_kinematics, trip_summary, WINDOW_SIZE_SECONDS


def _predominant(windows: pd.DataFrame) -> str | None:
    if windows.empty:
        return None
    w = windows.groupby("cluster_label")["duration_s"].sum()
    return str(w.idxmax())


def build_trip_figure(k: pd.DataFrame, windows: pd.DataFrame) -> go.Figure:
    t = k["elapsed_s"] / 60.0
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.28, 0.28, 0.32, 0.12],
        subplot_titles=("Velocidade (km/h)", "Aceleração longitudinal (m/s²)",
                        "Potência elétrica da bateria (kW)", "Perfil por janela de 120 s"),
    )
    fig.add_trace(go.Scatter(x=t, y=k["Vehicle Speed[km/h]"], name="Velocidade",
                             line=dict(color=SPEED_COLOR, width=1.6),
                             hovertemplate="%{y:.1f} km/h<extra></extra>"), row=1, col=1)

    a = k["accel_ms2"]
    fig.add_hrect(y0=2.0, y1=max(3.0, a.max() + 0.3), fillcolor=CONSUME_COLOR, opacity=0.12,
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=min(-3.0, a.min() - 0.3), y1=-2.0, fillcolor="#D64541", opacity=0.12,
                  line_width=0, row=2, col=1)
    for y, txt, col in [(2.0, "Arrancada rápida (≥ +2,0)", CONSUME_COLOR),
                        (-2.0, "Frenagem brusca (≤ −2,0)", "#D64541")]:
        fig.add_hline(y=y, line_dash="dash", line_color=col, row=2, col=1,
                      annotation_text=txt, annotation_position="top left", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=t, y=a, name="Aceleração", line=dict(color="#555", width=1.2),
                             hovertemplate="%{y:.2f} m/s²<extra></extra>"), row=2, col=1)
    ev = k[(a <= -2.0) | (a >= 2.0)]
    if not ev.empty:
        fig.add_trace(go.Scatter(
            x=ev["elapsed_s"] / 60.0, y=ev["accel_ms2"], mode="markers", name="Eventos bruscos",
            marker=dict(color=np.where(ev["accel_ms2"] > 0, CONSUME_COLOR, "#D64541"), size=7),
            hovertemplate="Evento: %{y:.2f} m/s²<extra></extra>"), row=2, col=1)

    p = k["power_kw"]
    fig.add_trace(go.Scatter(x=t, y=p.clip(lower=0), name="Consumo (P > 0)", fill="tozeroy",
                             line=dict(color=CONSUME_COLOR, width=1),
                             hovertemplate="%{y:.1f} kW<extra>Consumo</extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=p.clip(upper=0), name="Regeneração (P < 0)", fill="tozeroy",
                             line=dict(color=REGEN_COLOR, width=1),
                             hovertemplate="%{y:.1f} kW<extra>Regeneração</extra>"), row=3, col=1)

    shown = set()
    for _, w in windows.iterrows():
        label = w["cluster_label"]
        start = w["window_id"] * WINDOW_SIZE_SECONDS / 60.0
        fig.add_trace(go.Bar(
            x=[WINDOW_SIZE_SECONDS / 60.0], base=[start], y=["Janelas"], orientation="h",
            marker=dict(color=PROFILE_COLORS.get(label, "#999"), line=dict(color="white", width=1.5)),
            name=label, legendgroup=label, showlegend=label not in shown,
            hovertemplate=(f"Janela {int(w['window_id'])}: {label}<br>"
                           f"Vel. média {w['mean_speed_kmh']:.1f} km/h<extra></extra>"),
        ), row=4, col=1)
        shown.add(label)

    fig.update_xaxes(title_text="Tempo decorrido (min)", row=4, col=1)
    fig.update_yaxes(showticklabels=False, row=4, col=1)
    fig.update_layout(height=860, template="plotly_white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), barmode="overlay",
                      legend=dict(orientation="h", y=-0.08))
    return fig


def render(veh_id: int | None, trip_id: int | None):
    st.subheader("🚗 Inspetor Interativo de Viagens Reais (VED)")
    if veh_id is None or trip_id is None:
        st.info("Selecione um veículo e uma viagem na barra lateral.")
        return

    df = load_clusters()
    windows = df[(df["VehId"] == veh_id) & (df["Trip"] == trip_id)].sort_values("window_id")
    k = trip_kinematics(veh_id, trip_id)
    if k.empty:
        st.warning(f"Sem telemetria para Veículo {veh_id} · Viagem {trip_id}.")
        return

    s = trip_summary(k)
    pred = _predominant(windows)
    head_l, head_r = st.columns([3, 1.3])
    head_l.markdown(f"**Veículo {veh_id}** (Nissan Leaf 2013) · **Viagem {trip_id}** · "
                    f"{len(windows)} janela(s) de 120 s")
    if pred:
        head_r.markdown(f"<span class='badge' style='background:{PROFILE_COLORS[pred]}'>"
                        f"{PROFILE_ICONS[pred]} Predominante: {pred}</span>", unsafe_allow_html=True)

    c = st.columns(6)
    m, sec = divmod(int(s["duration_s"]), 60)
    c[0].metric("Duração", f"{m} min {sec:02d} s")
    c[1].metric("Distância", f"{s['distance_km']:.2f} km")
    c[2].metric("Vel. média", f"{s['mean_speed']:.1f} km/h", f"máx {s['max_speed']:.0f} km/h",
                delta_color="off")
    c[3].metric("Energia líquida", f"{s['net_kwh']:.3f} kWh",
                f"regen {s['regen_kwh']:.3f} kWh", delta_color="off")
    c[4].metric("Consumo específico",
                "—" if np.isnan(s["wh_per_km"]) else f"{s['wh_per_km']:.0f} Wh/km")
    c[5].metric("Freadas / Arrancadas", f"{s['hard_brakes']} / {s['rapid_accels']}")

    if windows.empty:
        st.caption("Viagem sem janelas válidas (≥ 30 s) na base rotulada.")
    else:
        dist = windows["cluster_label"].value_counts()
        st.caption(" · ".join(f"{PROFILE_ICONS[l]} {l}: {int(dist.get(l, 0))}" for l in PROFILE_ORDER))

    st.plotly_chart(build_trip_figure(k, windows), use_container_width=True)

    with st.expander("📋 Tabela detalhada das janelas de 120 s"):
        cols = ["window_id", "cluster_label", "duration_s", "mean_speed_kmh", "max_speed_kmh",
                "std_speed_kmh", "max_pos_accel_ms2", "hard_braking_events", "rapid_accel_events",
                "mean_power_kw", "total_energy_kwh"]
        st.dataframe(windows[cols], use_container_width=True, hide_index=True)
