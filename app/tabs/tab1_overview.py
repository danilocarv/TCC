"""Aba 1 — Visão Geral dos Perfis e Métricas dos Clusters."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import PROFILE_COLORS, PROFILE_ORDER, PROFILE_ICONS, FEATURE_NAMES
from app.data_service import load_clusters, load_metrics, centroids_pca, CLUSTER_FEATURE_COLS


def _fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def build_pca_figure(df: pd.DataFrame, opacity: float = 0.6, show_centroids: bool = True) -> go.Figure:
    metrics = load_metrics()
    var = metrics.get("pca_explained_variance_ratio", [0, 0])
    fig = go.Figure()
    for label in PROFILE_ORDER:
        sub = df[df["cluster_label"] == label]
        fig.add_trace(go.Scattergl(
            x=sub["pca_x"], y=sub["pca_y"], mode="markers",
            name=f"{PROFILE_ICONS[label]} {label} ({len(sub)})",
            marker=dict(color=PROFILE_COLORS[label], size=6, opacity=opacity),
            customdata=sub[["VehId", "Trip", "window_id", "mean_speed_kmh",
                            "hard_braking_rate_min", "rapid_accel_rate_min"]].values,
            hovertemplate=(
                "<b>Veículo %{customdata[0]}</b> · Viagem %{customdata[1]} · Janela %{customdata[2]}<br>"
                "Velocidade média: %{customdata[3]:.1f} km/h<br>"
                "Freadas bruscas: %{customdata[4]:.2f}/min<br>"
                "Arrancadas: %{customdata[5]:.2f}/min<extra>" + label + "</extra>"
            ),
        ))
    if show_centroids:
        c = centroids_pca()
        fig.add_trace(go.Scatter(
            x=c[:, 0], y=c[:, 1], mode="markers+text", name="Centróides",
            text=["Eco", "Mod", "Agr"], textposition="top center",
            marker=dict(symbol="x", size=16, color="black", line=dict(width=2, color="white")),
            hovertemplate="Centróide %{text}<br>PC1=%{x:.2f} · PC2=%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=f"PC1 ({var[0]*100:.1f}% da variância)",
        yaxis_title=f"PC2 ({var[1]*100:.1f}% da variância)",
        height=540, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def render():
    df = load_clusters()
    metrics = load_metrics()
    ev = metrics["evaluation_metrics"]
    i3 = ev["k"].index(metrics.get("optimal_k_selected", 3))

    st.subheader("📊 Visão Geral dos Perfis de Condução (K-Means, k = 3)")
    c = st.columns(5)
    c[0].metric("Total de janelas (120 s)", _fmt(len(df)))
    c[1].metric("Janelas ativas (v ≥ 8 km/h)", _fmt((df["mean_speed_kmh"] >= 8.0).sum()))
    c[2].metric("Coef. de Silhueta", f"{ev['silhouette'][i3]:.3f}", help="Maior é melhor (−1 a 1)")
    c[3].metric("Calinski-Harabasz", f"{ev['calinski_harabasz'][i3]:.1f}", help="Maior é melhor")
    c[4].metric("Davies-Bouldin", f"{ev['davies_bouldin'][i3]:.2f}", help="Menor é melhor")

    counts = df["cluster_label"].value_counts()
    cols = st.columns(3)
    for col, label in zip(cols, PROFILE_ORDER):
        n = int(counts.get(label, 0))
        col.markdown(
            f"<div class='profile-card' style='background:{PROFILE_COLORS[label]}'>"
            f"<h3>{PROFILE_ICONS[label]} {label}</h3>{n/len(df)*100:.1f}% · {_fmt(n)} janelas</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Espaço latente PCA 2D")
        var = metrics.get("pca_explained_variance_ratio", [0, 0])
        st.caption(f"Variância explicada total: {sum(var)*100:.1f}%. Passe o mouse sobre os pontos.")
        st.plotly_chart(build_pca_figure(df), use_container_width=True)

    with right:
        st.markdown("#### Comparativo dos centróides")
        mode = st.radio("Visualização", ["Barras (normalizado)", "Radar (normalizado)", "Tabela (valores reais)"],
                        horizontal=True, label_visibility="collapsed")
        cent = df.groupby("cluster_label")[CLUSTER_FEATURE_COLS].mean().reindex(PROFILE_ORDER)
        norm = (cent - cent.min()) / (cent.max() - cent.min()).replace(0, 1)
        names = [FEATURE_NAMES[f] for f in CLUSTER_FEATURE_COLS]

        fig = go.Figure()
        if mode.startswith("Barras"):
            for label in PROFILE_ORDER:
                fig.add_trace(go.Bar(
                    x=names, y=norm.loc[label], name=label, marker_color=PROFILE_COLORS[label],
                    customdata=cent.loc[label].values,
                    hovertemplate="%{x}<br>Valor real: %{customdata:.3f}<extra>" + label + "</extra>",
                ))
            fig.update_layout(barmode="group", yaxis_title="Intensidade relativa (0–1)", xaxis_tickangle=-25)
        elif mode.startswith("Radar"):
            for label in PROFILE_ORDER:
                fig.add_trace(go.Scatterpolar(
                    r=list(norm.loc[label]) + [norm.loc[label].iloc[0]],
                    theta=names + [names[0]], name=label, fill="toself", opacity=0.55,
                    line_color=PROFILE_COLORS[label],
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1.05])))
        if mode.startswith("Tabela"):
            st.dataframe(cent.rename(columns=FEATURE_NAMES).T.style.format("{:.3f}"), use_container_width=True)
        else:
            fig.update_layout(height=500, template="plotly_white", legend=dict(orientation="h", y=-0.3),
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
