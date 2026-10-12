"""
Dashboard Interativo do TCC — Perfis de Condução em Veículos Elétricos (VED).

Autor: Danilo Carvalho de Oliveira
Orientador: Prof. Me. Douglas Donizeti de Castilho Braz
IFSULDEMINAS - Campus Poços de Caldas

Execução:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="TCC · Perfis de Condução em EVs", page_icon="⚡", layout="wide")

from app.config import CUSTOM_CSS, PROFILE_COLORS, PROFILE_ICONS, PROFILE_ORDER  # noqa: E402
from app import data_service as ds  # noqa: E402
from app.tabs import tab1_overview, tab2_inspector, tab3_simulator  # noqa: E402

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="tcc-header">
      <h1>⚡ Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos</h1>
      <p><b>Autor:</b> Danilo Carvalho de Oliveira &nbsp;·&nbsp; <b>Orientador:</b> Prof. Me. Douglas Donizeti de Castilho Braz</p>
      <p>IFSULDEMINAS – Campus Poços de Caldas &nbsp;·&nbsp; Vehicle Energy Dataset (VED) – Nissan Leaf 2013 (IDs 10, 455, 541)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = ds.missing_files()
if missing:
    st.error("Artefatos obrigatórios não encontrados. Execute o pipeline da Fase 1:\n\n- " + "\n- ".join(missing))
    st.stop()

df_clusters = ds.load_clusters()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🚗 Inspetor de Viagens")
    vehicles = sorted(df_clusters["VehId"].unique().tolist())
    veh_id = st.selectbox("Veículo (VehId)", vehicles, format_func=lambda v: f"Veículo {v}", key="veh_sel")
    trips = ds.trips_for_vehicle(df_clusters, veh_id)
    win_counts = df_clusters[df_clusters["VehId"] == veh_id].groupby("Trip").size()
    trip_id = st.selectbox(
        f"Viagem (Trip) — {len(trips)} disponíveis", trips,
        format_func=lambda t: f"Trip {t} ({int(win_counts.get(t, 0))} janelas)",
        key=f"trip_sel_{veh_id}",
    ) if trips else None
    st.caption("A chave composta (VehId, Trip) identifica unicamente cada viagem.")

    st.divider()
    st.markdown("### Perfis identificados")
    counts = df_clusters["cluster_label"].value_counts()
    for label in PROFILE_ORDER:
        n = int(counts.get(label, 0))
        st.markdown(f"{PROFILE_ICONS[label]} **{label}** — {n/len(df_clusters)*100:.1f}% ({n})")
    st.divider()
    st.success("Dados e modelos carregados ✔")

tab1, tab2, tab3 = st.tabs([
    "📊 Visão Geral dos Perfis", "🚗 Inspetor de Viagens Reais", "🎮 Simulador de Condutores",
])
with tab1:
    tab1_overview.render()
with tab2:
    tab2_inspector.render(int(veh_id), int(trip_id) if trip_id is not None else None)
with tab3:
    tab3_simulator.render()
