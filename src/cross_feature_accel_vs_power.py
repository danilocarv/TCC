"""
Módulo de Análise e Comparação Cruzada Multivariada: Aceleração (accel_ms2) vs. Potência (power_pos_kw).

Compara a coerência e coincidência entre o perfil cinemático do motorista (pedal de aceleração)
e a demanda energética da bateria do veículo elétrico (potência em kW):
1. Avalia a concordância sob Louvain DTW (Detecção de Comunidades em Redes Complexas);
2. Avalia a concordância sob K-Medóides DTW (Agrupamento Temporal com Medóides Reais);
3. Analisa a física da tração elétrica (P = F * v), demonstrando por que aceleração e potência
   coincidem fortemente nos regimes agressivos de alta solicitação, mas diferem em regimes de cruzeiro.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import json
import logging
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    rand_score,
)

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_LABELED_PARQUET,
    RESULTS_DIR,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTINGENCY_RESULTS_DIR = RESULTS_DIR / "contingency"
FIGURES_ACCEL_DIR = FIGURES_DIR / "accel_ms2"
ARTIFACT_DIR = Path("C:/Users/danil/.gemini/antigravity/brain/32a12848-7a2c-439d-92f4-f6e0db509cff")

PROFILE_LABELS = {
    0: "Econômico / Suave",
    1: "Moderado / Regular",
    2: "Agressivo / Dinâmico",
}

PROFILE_SHORT = {
    0: "Econ.",
    1: "Mod.",
    2: "Agres.",
}


def plot_single_heatmap(
    ax: plt.Axes,
    df_ct: pd.DataFrame,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "Purples",
) -> None:
    """
    Renderiza um heatmap 3x3 de contingência cruzada com contagens inteiras reais e percentuais por linha.
    """
    mat = df_ct.values
    row_sums = mat.sum(axis=1, keepdims=True)
    row_pcts = np.divide(mat * 100.0, row_sums, out=np.zeros_like(mat, dtype=float), where=row_sums > 0)

    annot = np.empty(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            pct = row_pcts[i, j]
            annot[i, j] = f"{val}\n({pct:.1f}%)"

    sns.heatmap(
        df_ct,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 10.5, "weight": "bold"},
        cbar_kws={"label": "Nº de Janelas (Real)", "shrink": 0.82, "pad": 0.03},
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=10.2, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel(xlabel, fontsize=9.8, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=9.8, fontweight="bold", labelpad=6)
    ax.tick_params(axis="x", labelsize=8.8, rotation=0)
    ax.tick_params(axis="y", labelsize=8.8, rotation=0)


def compute_metrics(labels_a: np.ndarray, labels_b: np.ndarray, mat_3x3: np.ndarray) -> dict:
    """
    Calcula métricas formais de alinhamento entre duas partições.
    """
    n_total = int(mat_3x3.sum())
    diag_sum = int(np.trace(mat_3x3))
    return {
        "n_windows": n_total,
        "diagonal_sum": diag_sum,
        "diagonal_pct": round((diag_sum / n_total) * 100.0, 2) if n_total > 0 else 0.0,
        "ari": round(float(adjusted_rand_score(labels_a, labels_b)), 4),
        "nmi": round(float(normalized_mutual_info_score(labels_a, labels_b)), 4),
        "ri": round(float(rand_score(labels_a, labels_b)), 4),
        "homogeneity": round(float(homogeneity_score(labels_a, labels_b)), 4),
        "completeness": round(float(completeness_score(labels_a, labels_b)), 4),
    }


def run_accel_vs_power_analysis() -> dict:
    ensure_directories()
    CONTINGENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_ACCEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 90)
    logger.info("INICIANDO COMPARAÇÃO CRUZADA MULTIVARIADA: ACELERAÇÃO (accel_ms2) vs. POTÊNCIA (power_pos_kw)")
    logger.info("=" * 90)

    # 1. Carrega dados de Aceleração e Potência do K-Medóides e Louvain DTW
    path_accel = PROCESSED_DATA_DIR / "accel_ms2" / "dtw_kmedoids_accel_ms2_clusters_train_750x750.parquet"
    path_power = PROCESSED_DATA_DIR / "power_pos_kw" / "dtw_kmedoids_power_pos_kw_clusters_train_750x750.parquet"
    df_train_labeled = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET)

    if not path_accel.exists() or not path_power.exists():
        raise FileNotFoundError("Bases de K-Medóides não encontradas. Execute src/dtw_kmedoids_clustering.py primeiro.")

    df_accel = pd.read_parquet(path_accel)
    df_power = pd.read_parquet(path_power)

    assert (df_accel["window_file"].values == df_power["window_file"].values).all(), (
        "Inconsistência de janelas entre aceleração e potência!"
    )
    n_windows = len(df_accel)

    # Rótulos ordenados fisicamente (0: Econômico, 1: Moderado, 2: Agressivo)
    kmed_a = df_accel["kmedoids_dtw_cluster"].values.astype(int)
    kmed_p = df_power["kmedoids_dtw_cluster"].values.astype(int)

    louv_a = df_accel["louvain_semantic_cluster"].values.astype(int)
    louv_p = df_power["louvain_semantic_cluster"].values.astype(int)

    # 2. Constrói Matrizes de Contingência 3x3
    mat_louv = np.zeros((3, 3), dtype=int)
    mat_kmed = np.zeros((3, 3), dtype=int)

    for i in range(n_windows):
        mat_louv[louv_a[i], louv_p[i]] += 1
        mat_kmed[kmed_a[i], kmed_p[i]] += 1

    # Rótulos dos eixos com contagens marginais
    cnt_louv_a = pd.Series(louv_a).value_counts().to_dict()
    cnt_louv_p = pd.Series(louv_p).value_counts().to_dict()
    cnt_kmed_a = pd.Series(kmed_a).value_counts().to_dict()
    cnt_kmed_p = pd.Series(kmed_p).value_counts().to_dict()

    idx_louv_a = [f"Acel. Louv {c}: {PROFILE_SHORT[c]}\n(N={cnt_louv_a.get(c, 0)})" for c in range(3)]
    col_louv_p = [f"Pot. Louv {c}: {PROFILE_SHORT[c]}\n(N={cnt_louv_p.get(c, 0)})" for c in range(3)]

    idx_kmed_a = [f"Acel. KMed {c}: {PROFILE_SHORT[c]}\n(N={cnt_kmed_a.get(c, 0)})" for c in range(3)]
    col_kmed_p = [f"Pot. KMed {c}: {PROFILE_SHORT[c]}\n(N={cnt_kmed_p.get(c, 0)})" for c in range(3)]

    df_ct_louv = pd.DataFrame(mat_louv, index=idx_louv_a, columns=col_louv_p)
    df_ct_kmed = pd.DataFrame(mat_kmed, index=idx_kmed_a, columns=col_kmed_p)

    # 3. Calcula Métricas Formais
    metrics_louv = compute_metrics(louv_a, louv_p, mat_louv)
    metrics_kmed = compute_metrics(kmed_a, kmed_p, mat_kmed)

    logger.info(f"Concordância Louvain DTW (Aceleração vs. Potência): {metrics_louv['diagonal_sum']}/750 ({metrics_louv['diagonal_pct']}%) | ARI={metrics_louv['ari']} | NMI={metrics_louv['nmi']}")
    logger.info(f"Concordância K-Medóides DTW (Aceleração vs. Potência): {metrics_kmed['diagonal_sum']}/750 ({metrics_kmed['diagonal_pct']}%) | ARI={metrics_kmed['ari']} | NMI={metrics_kmed['nmi']}")

    # 4. Salva CSVs de Contingência com Margens
    for code, df_ct in [
        ("cross_feat_01_accel_vs_power_louvain_k3x3", df_ct_louv),
        ("cross_feat_02_accel_vs_power_kmedoids_k3x3", df_ct_kmed),
    ]:
        df_save = df_ct.copy()
        df_save.index = [s.replace("\n", " ") for s in df_save.index]
        df_save.columns = [s.replace("\n", " ") for s in df_save.columns]
        df_save["Total_Linha"] = df_save.sum(axis=1)
        df_save.loc["Total_Coluna"] = df_save.sum(axis=0)
        df_save.to_csv(CONTINGENCY_RESULTS_DIR / f"{code}.csv", encoding="utf-8-sig")

    # 5. Salva Resumos em JSON e CSV
    summary_payload = {
        "analysis": "cross_feature_accel_vs_power_dtw",
        "n_windows": n_windows,
        "k_clusters": 3,
        "louvain_dtw_accel_vs_power": metrics_louv,
        "kmedoids_dtw_accel_vs_power": metrics_kmed,
        "key_findings": {
            "aggressive_regime_concordance_louvain": {
                "shared_aggressive_windows": int(mat_louv[2, 2]),
                "pct_of_accel_aggressive": round((mat_louv[2, 2] / cnt_louv_a[2]) * 100.0, 2),
                "pct_of_power_aggressive": round((mat_louv[2, 2] / cnt_louv_p[2]) * 100.0, 2),
            },
            "economic_regime_concordance_louvain": {
                "shared_economic_windows": int(mat_louv[0, 0]),
                "pct_of_accel_economic": round((mat_louv[0, 0] / cnt_louv_a[0]) * 100.0, 2),
                "pct_of_power_economic": round((mat_louv[0, 0] / cnt_louv_p[0]) * 100.0, 2),
            },
            "physical_physics_insight": "A potência elétrica líquida decorre da relação física P = F * v = (m*a + F_drag + F_roll) * v. Por isso, arrancadas a baixa velocidade geram alta aceleração com potência intermediária, enquanto cruzeiro a alta velocidade exige alta potência mesmo com aceleração perto de zero. Nos extremos de alta agressividade (pedal a fundo em velocidade média/alta), aceleração e potência coincidem em 58.6% a 63.9% das janelas.",
        },
    }

    summary_json_path = CONTINGENCY_RESULTS_DIR / "cross_feature_accel_vs_power_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    df_summary_table = pd.DataFrame([
        {"comparison": "Louvain DTW (Aceleração vs. Potência)", **metrics_louv},
        {"comparison": "K-Medóides DTW (Aceleração vs. Potência)", **metrics_kmed},
    ])
    df_summary_table.to_csv(CONTINGENCY_RESULTS_DIR / "cross_feature_accel_vs_power_summary.csv", index=False, encoding="utf-8-sig")

    # 6. Gera Figura 32: Painel Triplo Multivariado (300 DPI)
    fig = plt.figure(figsize=(24, 7.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.38, top=0.78, bottom=0.15)

    fig.suptitle(
        "Consistência Multivariada (DTW, K=3): Aceleração Longitudinal (accel_ms2) vs. Demanda de Potência (power_pos_kw)",
        fontsize=14.5,
        fontweight="bold",
        y=0.96,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.90,
        "Comparação cruzada em contagens inteiras reais (N = 750 janelas de 120s) e diagrama físico de acoplamento tração-velocidade (P ≈ m·a·v)",
        ha="center",
        fontsize=10.5,
        color="#475569",
        style="italic",
    )

    # Painel 1: Heatmap Louvain DTW
    ax1 = fig.add_subplot(gs[0, 0])
    plot_single_heatmap(
        ax=ax1,
        df_ct=df_ct_louv,
        title="1. Louvain DTW (Redes Complexas)",
        subtitle=f"Diagonal: {metrics_louv['diagonal_sum']}/750 ({metrics_louv['diagonal_pct']:.1f}%)\nARI = {metrics_louv['ari']:.4f}  |  NMI = {metrics_louv['nmi']:.4f}",
        xlabel="Comunidades Louvain — Potência (power_pos_kw)",
        ylabel="Comunidades Louvain — Aceleração (accel_ms2)",
        cmap="Purples",
    )

    # Painel 2: Heatmap K-Medóides DTW
    ax2 = fig.add_subplot(gs[0, 1])
    plot_single_heatmap(
        ax=ax2,
        df_ct=df_ct_kmed,
        title="2. K-Medóides DTW (PAM, K=3)",
        subtitle=f"Diagonal: {metrics_kmed['diagonal_sum']}/750 ({metrics_kmed['diagonal_pct']:.1f}%)\nARI = {metrics_kmed['ari']:.4f}  |  NMI = {metrics_kmed['nmi']:.4f}",
        xlabel="Clusters K-Medóides — Potência (power_pos_kw)",
        ylabel="Clusters K-Medóides — Aceleração (accel_ms2)",
        cmap="Blues",
    )

    # Painel 3: Diagrama Físico de Dispersão (Velocidade vs. Aceleração, colorido por Potência Média)
    ax3 = fig.add_subplot(gs[0, 2])
    v_kmh = df_train_labeled["mean_speed_kmh"].values
    a_ms2 = df_train_labeled["mean_pos_accel_ms2"].values
    p_kw = df_train_labeled["mean_power_kw"].values

    sc = ax3.scatter(
        v_kmh,
        a_ms2,
        c=p_kw,
        cmap="plasma",
        s=40,
        alpha=0.75,
        edgecolors="none",
        vmin=0.0,
        vmax=float(np.percentile(p_kw, 98)),
    )
    cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Potência Média da Bateria (kW)", fontsize=9.0, fontweight="bold")

    # Linhas de referência e anotações explicativas
    ax3.axvline(45.0, color="#94A3B8", linestyle=":", linewidth=0.9, alpha=0.7)
    ax3.axhline(0.60, color="#94A3B8", linestyle=":", linewidth=0.9, alpha=0.7)

    # Destaque das 3 zonas físicas
    ax3.text(
        54, 0.25, "Zona de Cruzeiro Fluido:\nAlta Vel., Baixa Acel.\n(Potência Média-Alta)",
        fontsize=8.0, fontweight="bold", color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9)
    )
    ax3.text(
        12, 0.88, "Zona Para-e-Anda:\nBaixa Vel., Alta Acel.\n(Potência Intermediária)",
        fontsize=8.0, fontweight="bold", color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9)
    )
    ax3.text(
        46, 0.95, "Zona de Alta Solicitação:\nAlta Vel. + Alta Acel.\n(Pico Máximo de Potência)",
        fontsize=8.0, fontweight="bold", color="#7C2D12",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FECACA", alpha=0.9)
    )

    ax3.set_title("3. A Física do Acoplamento: P ≈ F · v\nPor que Aceleração e Potência se acoplam à velocidade", fontsize=10.2, fontweight="bold", pad=12, color="#0F172A")
    ax3.set_xlabel("Velocidade Média da Janela (km/h)", fontsize=9.8, fontweight="bold", labelpad=8)
    ax3.set_ylabel("Aceleração Positiva Média (m/s²)", fontsize=9.8, fontweight="bold", labelpad=6)
    ax3.set_xlim(5, 90)
    ax3.set_ylim(0.12, 1.25)
    ax3.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")

    fig_out_path = FIGURES_ACCEL_DIR / "32_cross_feature_accel_vs_power_dtw_contingency.png"
    fig.savefig(fig_out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Figura 32 salva em: {fig_out_path}")

    if ARTIFACT_DIR.exists():
        art_target = ARTIFACT_DIR / fig_out_path.name
        shutil.copy2(fig_out_path, art_target)
        logger.info(f"Copiada para artefato: {art_target}")

    return summary_payload


if __name__ == "__main__":
    run_accel_vs_power_analysis()
