"""
Camada de dados do dashboard: leitura cacheada de Parquet/JSON, carregamento
dos modelos .joblib e cálculo de cinemática/estatísticas por viagem (VehId, Trip).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    PROCESSED_CLUSTERS_PARQUET,
    RAW_EV_PARQUET,
    KMEANS_MODEL_PATH,
    SCALER_MODEL_PATH,
    PCA_MODEL_PATH,
    CLUSTERING_METRICS_JSON,
    CLUSTER_FEATURE_COLS,
    WINDOW_SIZE_SECONDS,
)
from src.feature_engineering import compute_point_kinematics  # noqa: E402

REQUIRED_FILES = {
    "Base rotulada (clusters)": PROCESSED_CLUSTERS_PARQUET,
    "Telemetria 1 Hz": RAW_EV_PARQUET,
    "Modelo K-Means": KMEANS_MODEL_PATH,
    "Scaler": SCALER_MODEL_PATH,
    "PCA": PCA_MODEL_PATH,
    "Métricas": CLUSTERING_METRICS_JSON,
}


def missing_files() -> list[str]:
    """Lista os artefatos obrigatórios ausentes em disco."""
    return [f"{name}: {path}" for name, path in REQUIRED_FILES.items() if not Path(path).exists()]


@st.cache_data(show_spinner="Carregando janelas rotuladas...")
def load_clusters() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_CLUSTERS_PARQUET)
    df["VehId"] = df["VehId"].astype(int)
    df["Trip"] = df["Trip"].astype(int)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(0.0)
    df["cluster_label"] = df["cluster_label"].fillna("Moderado / Regular")
    return df


@st.cache_data(show_spinner="Carregando telemetria 1 Hz...")
def load_telemetry() -> pd.DataFrame:
    df = pd.read_parquet(RAW_EV_PARQUET)
    df["VehId"] = df["VehId"].astype(int)
    df["Trip"] = df["Trip"].astype(int)
    return df


@st.cache_data
def load_metrics() -> dict:
    with open(CLUSTERING_METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Carregando modelos de ML...")
def load_models():
    """Carrega (modelo, scaler, pca). Injeta DrivingProfileModel em __main__,
    pois o wrapper foi serializado quando src/clustering.py rodava como script."""
    import __main__
    from src.clustering import DrivingProfileModel

    if not hasattr(__main__, "DrivingProfileModel"):
        __main__.DrivingProfileModel = DrivingProfileModel
    model = joblib.load(KMEANS_MODEL_PATH)
    scaler = joblib.load(SCALER_MODEL_PATH)
    pca = joblib.load(PCA_MODEL_PATH)
    return model, scaler, pca


def trips_for_vehicle(df_clusters: pd.DataFrame, veh_id: int) -> list[int]:
    return sorted(df_clusters.loc[df_clusters["VehId"] == veh_id, "Trip"].unique().tolist())


@st.cache_data(show_spinner="Processando viagem...")
def trip_kinematics(veh_id: int, trip_id: int) -> pd.DataFrame:
    """Telemetria da viagem (VehId, Trip) regularizada a 1 Hz com aceleração e potência."""
    tel = load_telemetry()
    raw = tel[(tel["VehId"] == veh_id) & (tel["Trip"] == trip_id)]
    if raw.empty:
        return pd.DataFrame()
    with contextlib.redirect_stdout(io.StringIO()):  # silencia logs do pipeline
        k = compute_point_kinematics(raw)
    k["elapsed_s"] = k["second_id"] - k["second_id"].min()
    k["window_id"] = (k["elapsed_s"] // WINDOW_SIZE_SECONDS).astype(int)
    for col in ["Vehicle Speed[km/h]", "accel_ms2", "power_kw"]:
        k[col] = k[col].fillna(0.0)
    return k


def trip_summary(k: pd.DataFrame) -> dict:
    dt = k["dt"]
    p = k["power_kw"]
    dist_km = float((k["v_ms"] * dt).sum() / 1000.0)
    consumed = float((p.clip(lower=0) * dt).sum() / 3600.0)
    regen = float((p.clip(upper=0) * dt).sum() / 3600.0)
    net = consumed + regen
    return {
        "duration_s": float(k["elapsed_s"].max()),
        "distance_km": dist_km,
        "mean_speed": float(k["Vehicle Speed[km/h]"].mean()),
        "max_speed": float(k["Vehicle Speed[km/h]"].max()),
        "consumed_kwh": consumed,
        "regen_kwh": regen,
        "net_kwh": net,
        "wh_per_km": net * 1000.0 / dist_km if dist_km > 0.05 else float("nan"),
        "hard_brakes": int((k["accel_ms2"] <= -2.0).sum()),
        "rapid_accels": int((k["accel_ms2"] >= 2.0).sum()),
    }


def predict_profile(features: dict):
    """Classifica um vetor de 6 atributos. Retorna (id, rótulo, distâncias, ponto PCA)."""
    model, scaler, pca = load_models()
    x = np.array([[float(features[c]) for c in CLUSTER_FEATURE_COLS]])
    xs = scaler.transform(x)
    cid = int(model.predict(xs)[0])
    label = str(model.predict_label(xs)[0])
    dists = model.transform(xs)[0]
    pt = pca.transform(xs)[0]
    return cid, label, dists, pt


@st.cache_data
def centroids_pca() -> np.ndarray:
    model, _, pca = load_models()
    return pca.transform(model.cluster_centers_)
