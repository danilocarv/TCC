"""
Módulo de Agrupamento K-DBA (TimeSeriesKMeans com DTW Barycenter Averaging via tslearn)
e Triangulação Metodológica Quádrupla.

Atende à recomendação do orientador Prof. Douglas Braz de aplicar a técnica clássica de
K-Means adaptada para séries temporais usando baricentro ótimo via DTW (DBA - Petitjean et al., 2011),
preservando a morfologia e a intensidade dos picos das curvas temporais sem achatamento euclidiano.

Estabelece a triangulação metodológica quádrupla (K = 3 perfis: Econômico, Moderado, Agressivo):
1. K-Means Baseline (Atributos Escalares Agregados + Distância Euclidiana);
2. Louvain DTW (Grafo Completo de Similaridade DTW + Modularidade de Newman-Girvan);
3. K-Medóides DTW (Série 1 Hz Pura + Distância DTW com Medóides Reais Auditáveis);
4. K-DBA DTW (Série 1 Hz Pura + TimeSeriesKMeans com Baricentros Sintéticos Médios).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
import warnings

# Suprime avisos do h5py do tslearn
warnings.filterwarnings("ignore", category=UserWarning, module="tslearn")

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
from tslearn.clustering import TimeSeriesKMeans

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_LABELED_PARQUET,
    RESULTS_DIR,
    WINDOWS_TRAIN_PARQUET_DIR,
    ensure_directories,
    get_feature_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTINGENCY_RESULTS_DIR = RESULTS_DIR / "contingency"
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

PROFILE_COLORS = {
    0: "#10B981",  # Verde esmeralda (Econômico)
    1: "#3B82F6",  # Azul royal (Moderado)
    2: "#EF4444",  # Vermelho vibrante (Agressivo)
}


def load_window_series_fast(feature: str = "accel_ms2", n_windows: int = 750) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega as séries temporais brutas (1 Hz, 120s) das janelas de treino."""
    fcfg = get_feature_paths(feature)
    col_name = fcfg["column"]
    clip_positive = fcfg.get("clip_positive", False)

    train_files = sorted(WINDOWS_TRAIN_PARQUET_DIR.glob("*.parquet"))[:n_windows]
    if len(train_files) < n_windows:
        raise FileNotFoundError(
            f"Esperado {n_windows} arquivos em {WINDOWS_TRAIN_PARQUET_DIR}, encontrado {len(train_files)}."
        )

    max_len = 120
    data_matrix = np.zeros((n_windows, max_len), dtype=np.float64)
    lengths = np.zeros(n_windows, dtype=np.int64)
    stems = []

    for idx, fpath in enumerate(train_files):
        df_w = pd.read_parquet(fpath, columns=[col_name])
        vals = df_w[col_name].to_numpy(dtype=np.float64)
        if clip_positive:
            vals = np.maximum(vals, 0.0)

        seq_len = min(len(vals), max_len)
        data_matrix[idx, :seq_len] = vals[:seq_len]
        lengths[idx] = seq_len
        stems.append(fpath.stem)

    return data_matrix, lengths, stems


def order_louvain_by_intensity(
    louv_raw: np.ndarray,
    df_train: pd.DataFrame,
    feature: str = "accel_ms2",
) -> tuple[np.ndarray, dict[int, int], dict[int, str]]:
    """Mapeia comunidades Louvain para ordem física de intensidade."""
    sort_col_map = {
        "accel_ms2": "mean_pos_accel_ms2",
        "speed_kmh": "mean_speed_kmh",
        "power_pos_kw": "mean_power_kw",
    }
    sort_col = sort_col_map.get(feature, "mean_pos_accel_ms2")

    unique_comms = sorted(np.unique(louv_raw))
    comm_stats = []
    for cid in unique_comms:
        mask = louv_raw == cid
        val = float(df_train.loc[mask, sort_col].mean()) if mask.any() else 0.0
        comm_stats.append((int(cid), val, int(mask.sum())))

    comm_stats.sort(key=lambda x: x[1])
    orig_to_sem = {orig_cid: sem_idx for sem_idx, (orig_cid, _, _) in enumerate(comm_stats)}
    sem_labels = np.array([orig_to_sem[int(c)] for c in louv_raw], dtype=int)

    label_names = {}
    for sem_idx, (orig_cid, _, count) in enumerate(comm_stats):
        short_prof = PROFILE_SHORT.get(sem_idx, f"G{sem_idx}")
        label_names[sem_idx] = f"Louv C{orig_cid}: {short_prof}\n(N={count})"

    return sem_labels, orig_to_sem, label_names


def order_kdba_by_intensity(
    raw_labels: np.ndarray,
    raw_centers: np.ndarray,
    df_train: pd.DataFrame,
    feature: str = "accel_ms2",
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """
    Ordena os clusters do K-DBA de forma estritamente crescente pela intensidade física,
    garantindo: 0 = Econômico, 1 = Moderado, 2 = Agressivo.
    """
    sort_col_map = {
        "accel_ms2": "mean_pos_accel_ms2",
        "speed_kmh": "mean_speed_kmh",
        "power_pos_kw": "mean_power_kw",
    }
    sort_col = sort_col_map.get(feature, "mean_pos_accel_ms2")

    cluster_stats = []
    for cid in range(3):
        mask = raw_labels == cid
        val = float(df_train.loc[mask, sort_col].mean()) if mask.any() else 0.0
        cluster_stats.append((cid, val))

    # Ordena pelo valor médio crescente
    cluster_stats.sort(key=lambda x: x[1])
    raw_to_sem = {raw_cid: sem_idx for sem_idx, (raw_cid, _) in enumerate(cluster_stats)}
    
    sem_labels = np.array([raw_to_sem[int(c)] for c in raw_labels], dtype=int)
    
    # Reordena os centros baricêntricos
    ordered_centers = np.zeros_like(raw_centers)
    for raw_cid, sem_idx in raw_to_sem.items():
        ordered_centers[sem_idx] = raw_centers[raw_cid]

    return sem_labels, ordered_centers, raw_to_sem


def compute_alignment_metrics(labels_a: np.ndarray, labels_b: np.ndarray, contingency_mat: np.ndarray) -> dict:
    """Calcula métricas padronizadas de concordância e similaridade entre dois agrupamentos."""
    n_total = int(len(labels_a))
    diag_sum = int(np.trace(contingency_mat))
    diag_pct = float((diag_sum / n_total) * 100.0)

    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    ri = float(rand_score(labels_a, labels_b))
    homo = float(homogeneity_score(labels_a, labels_b))
    comp = float(completeness_score(labels_a, labels_b))

    return {
        "n_samples": n_total,
        "diagonal_sum": diag_sum,
        "diagonal_pct": round(diag_pct, 2),
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "rand_index": round(ri, 4),
        "homogeneity": round(homo, 4),
        "completeness": round(comp, 4),
    }


def plot_kdba_barycenters(
    data_matrix: np.ndarray,
    lengths: np.ndarray,
    cluster_centers: np.ndarray,
    kdba_labels: np.ndarray,
    df_train: pd.DataFrame,
    feature: str,
    fig_path: Path,
    kmedoids_medoids: np.ndarray | None = None,
) -> None:
    """
    Gera a Figura 33: Painel dos 3 Baricentros Médios DTW (DBA) Sintéticos
    (120 segundos, 1 Hz) com Faixa de Dispersão e Comparativo com os Medóides Reais.
    """
    fcfg = get_feature_paths(feature)
    feat_name = fcfg["name_pt"]
    unit = fcfg["unit"]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(16, 12.5), dpi=300)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.05], hspace=0.45, wspace=0.28)

    fig.suptitle(
        f"K-DBA com Alinhamento DTW ({feat_name}) — Baricentros Médios Sintéticos dos 3 Perfis (120s)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.938,
        "Cada curva sólida é o baricentro ótimo calculado pelo algoritmo DBA (Petitjean et al., 2011), preservando os picos sem achatamento euclidiano",
        ha="center",
        fontsize=10.5,
        color="#475569",
        style="italic",
    )

    all_vals = np.concatenate([data_matrix[i, :lengths[i]] for i in range(len(data_matrix))])
    y_min = min(float(np.min(cluster_centers)) * 1.15, -0.5 if feature == "accel_ms2" else 0.0)
    y_max = float(np.max(cluster_centers)) * 1.35

    t = np.arange(120)

    for c_idx in range(3):
        ax = fig.add_subplot(gs[c_idx, :])
        center_curve = cluster_centers[c_idx]

        cluster_mask = kdba_labels == c_idx
        n_members = int(cluster_mask.sum())
        pct_members = (n_members / len(kdba_labels)) * 100.0

        # Coleta todas as séries do cluster para calcular percentis 25% e 75%
        member_series = data_matrix[cluster_mask, :120]
        q25 = np.percentile(member_series, 25, axis=0)
        q75 = np.percentile(member_series, 75, axis=0)

        # Estatísticas físicas do cluster
        cl_mean_accel = float(df_train.loc[cluster_mask, "mean_pos_accel_ms2"].mean())
        cl_max_accel = float(df_train.loc[cluster_mask, "max_pos_accel_ms2"].mean())
        cl_mean_speed = float(df_train.loc[cluster_mask, "mean_speed_kmh"].mean())
        cl_rapid_rate = float(df_train.loc[cluster_mask, "rapid_accel_rate_min"].mean())

        color = PROFILE_COLORS[c_idx]
        prof_title = PROFILE_LABELS[c_idx]

        # Faixa de dispersão intra-cluster (P25 - P75)
        ax.fill_between(t, q25, q75, color=color, alpha=0.15, label="Intervalo Interquartil (P25-P75)")

        # Linha zero
        if feature == "accel_ms2":
            ax.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=0.8, alpha=0.7)

        # Curva do Baricentro Sintético K-DBA
        ax.plot(
            t,
            center_curve,
            color=color,
            linewidth=2.4,
            label=f"Baricentro Médio K-DBA ({prof_title})",
            zorder=4,
        )

        # Sobreposição do Medóide Real do K-Medóides se disponível
        if kmedoids_medoids is not None and len(kmedoids_medoids) == 3:
            m_idx = int(kmedoids_medoids[c_idx])
            m_series = data_matrix[m_idx, :120]
            ax.plot(
                t,
                m_series,
                color="#0F172A",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label=f"Medóide Real K-Medóides (Janela #{m_idx})",
                zorder=3,
            )

        ax.set_xlim(0, 119)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(f"{feat_name}\n[{unit}]", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Tempo na Janela [segundos] (Amostragem a 1 Hz)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")

        title_text = (
            f"Cluster {c_idx}: {prof_title} — "
            f"Tamanho: N = {n_members} janelas ({pct_members:.1f}% da base)"
        )
        ax.set_title(title_text, fontsize=11, fontweight="bold", loc="left", pad=7, color="#0F172A")

        stats_box = (
            f"Estatísticas Físicas do Cluster {c_idx}:\n"
            f"• Aceleração Média (+): {cl_mean_accel:.3f} m/s²\n"
            f"• Aceleração Máxima:    {cl_max_accel:.3f} m/s²\n"
            f"• Velocidade Média:     {cl_mean_speed:.1f} km/h\n"
            f"• Eventos Bruscos/min:  {cl_rapid_rate:.2f}"
        )
        ax.text(
            0.985,
            0.92,
            stats_box,
            transform=ax.transAxes,
            fontsize=8.5,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor=color, alpha=0.92, linewidth=1.2),
            zorder=5,
        )
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Painel Inferior: Gráficos de Barras com Médias Físicas Comparativas dos 3 Grupos K-DBA
    clusters_x = np.arange(3)
    bar_width = 0.55
    bar_colors = [PROFILE_COLORS[i] for i in range(3)]
    bar_labels = [f"C{i}: {PROFILE_SHORT[i]}" for i in range(3)]

    # Subplot 1: Aceleração Média Positiva
    ax_b1 = fig.add_subplot(gs[3, 0])
    vals_b1 = [float(df_train.loc[kdba_labels == i, "mean_pos_accel_ms2"].mean()) for i in range(3)]
    bars1 = ax_b1.bar(clusters_x, vals_b1, width=bar_width, color=bar_colors, edgecolor="#0F172A", linewidth=0.8)
    ax_b1.set_xticks(clusters_x)
    ax_b1.set_xticklabels(bar_labels, fontsize=9.5, fontweight="semibold")
    ax_b1.set_ylabel("Acel. Média (+) [m/s²]", fontsize=9.5, fontweight="semibold")
    ax_b1.set_title("Aceleração Positiva Média", fontsize=10.5, fontweight="bold")
    ax_b1.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        h = bar.get_height()
        ax_b1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_b1.set_ylim(0, max(vals_b1) * 1.25)

    # Subplot 2: Velocidade Média
    ax_b2 = fig.add_subplot(gs[3, 1])
    vals_b2 = [float(df_train.loc[kdba_labels == i, "mean_speed_kmh"].mean()) for i in range(3)]
    bars2 = ax_b2.bar(clusters_x, vals_b2, width=bar_width, color=bar_colors, edgecolor="#0F172A", linewidth=0.8)
    ax_b2.set_xticks(clusters_x)
    ax_b2.set_xticklabels(bar_labels, fontsize=9.5, fontweight="semibold")
    ax_b2.set_ylabel("Velocidade Média [km/h]", fontsize=9.5, fontweight="semibold")
    ax_b2.set_title("Velocidade Média da Via", fontsize=10.5, fontweight="bold")
    ax_b2.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        ax_b2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_b2.set_ylim(0, max(vals_b2) * 1.25)

    # Subplot 3: Eventos Bruscos por Minuto
    ax_b3 = fig.add_subplot(gs[3, 2])
    vals_b3 = [float(df_train.loc[kdba_labels == i, "rapid_accel_rate_min"].mean()) for i in range(3)]
    bars3 = ax_b3.bar(clusters_x, vals_b3, width=bar_width, color=bar_colors, edgecolor="#0F172A", linewidth=0.8)
    ax_b3.set_xticks(clusters_x)
    ax_b3.set_xticklabels(bar_labels, fontsize=9.5, fontweight="semibold")
    ax_b3.set_ylabel("Arrancadas Bruscas/min", fontsize=9.5, fontweight="semibold")
    ax_b3.set_title("Taxa de Eventos Bruscos", fontsize=10.5, fontweight="bold")
    ax_b3.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in bars3:
        h = bar.get_height()
        ax_b3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_b3.set_ylim(0, max(max(vals_b3) * 1.25, 0.5))

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura 33 salva com sucesso: {fig_path}")


def plot_single_3x3_heatmap(
    ax: plt.Axes,
    df_ct: pd.DataFrame,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "Blues",
) -> None:
    """Plota um único mapa de calor 3x3 de contingência com anotações duplas (contagem e %)."""
    mat_core = df_ct.iloc[:3, :3].to_numpy(dtype=float)
    total_samples = int(df_ct.iloc[3, 3])

    annot_matrix = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            count = int(mat_core[r, c])
            pct_row = (count / df_ct.iloc[r, 3]) * 100.0 if df_ct.iloc[r, 3] > 0 else 0.0
            pct_tot = (count / total_samples) * 100.0
            annot_matrix[r, c] = f"{count}\n({pct_row:.1f}% lin\n{pct_tot:.1f}% tot)"

    x_labels = [f"{df_ct.columns[c]}\n[N={int(df_ct.iloc[3, c])}]" for c in range(3)]
    y_labels = [f"{df_ct.index[r]}\n[N={int(df_ct.iloc[r, 3])}]" for r in range(3)]

    sns.heatmap(
        mat_core,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        cbar=False,
        ax=ax,
        linewidths=1.2,
        linecolor="#E2E8F0",
        annot_kws={"fontsize": 9.2, "fontweight": "medium"},
    )

    ax.set_xticklabels(x_labels, fontsize=9.2, rotation=0, ha="center")
    ax.set_yticklabels(y_labels, fontsize=9.2, rotation=0, va="center")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold", labelpad=8)

    ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold", pad=10, color="#0F172A")


def plot_triangulation_panel(
    df_ct_kdba_kmed: pd.DataFrame,
    df_ct_kdba_louv: pd.DataFrame,
    df_ct_kdba_km: pd.DataFrame,
    metrics_dict: dict,
    feature: str,
    fig_path: Path,
) -> None:
    """Gera a Figura 34: Painel de Triangulação de Contingência 3x3 para o K-DBA."""
    fcfg = get_feature_paths(feature)
    feat_name = fcfg["name_pt"]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.9

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=300)

    fig.suptitle(
        f"Triangulação Metodológica ({feat_name}, K=3) — K-DBA (DTW Barycenter) vs. K-Medóides vs. Louvain vs. K-Means",
        fontsize=14,
        fontweight="bold",
        y=0.99,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.935,
        "Validação cruzada dos agrupamentos: diagonal principal indica concordância exata de classificação das 750 janelas",
        ha="center",
        fontsize=10.5,
        color="#475569",
        style="italic",
    )

    # Painel 1: K-DBA vs. K-Medóides DTW
    m1 = metrics_dict["kdba_vs_kmedoids"]
    plot_single_3x3_heatmap(
        ax=axes[0],
        df_ct=df_ct_kdba_kmed,
        title="1. K-DBA vs. K-Medóides DTW (Séries 1 Hz)",
        subtitle=f"Diagonal: {m1['diagonal_sum']}/750 ({m1['diagonal_pct']:.1f}%)\nARI = {m1['ari']:.4f}  |  NMI = {m1['nmi']:.4f}",
        xlabel="Clusters K-Medóides DTW (PAM, K=3)",
        ylabel="Clusters K-DBA DTW (tslearn, K=3)",
        cmap="Purples",
    )

    # Painel 2: K-DBA vs. Louvain DTW
    m2 = metrics_dict["kdba_vs_louvain"]
    plot_single_3x3_heatmap(
        ax=axes[1],
        df_ct=df_ct_kdba_louv,
        title="2. K-DBA vs. Louvain DTW (Grafo Completo)",
        subtitle=f"Diagonal: {m2['diagonal_sum']}/750 ({m2['diagonal_pct']:.1f}%)\nARI = {m2['ari']:.4f}  |  NMI = {m2['nmi']:.4f}",
        xlabel="Comunidades Louvain DTW (K=3)",
        ylabel="Clusters K-DBA DTW (tslearn, K=3)",
        cmap="Greens",
    )

    # Painel 3: K-DBA vs. K-Means Baseline
    m3 = metrics_dict["kdba_vs_kmeans"]
    plot_single_3x3_heatmap(
        ax=axes[2],
        df_ct=df_ct_kdba_km,
        title="3. K-DBA vs. K-Means Baseline (Tabular)",
        subtitle=f"Diagonal: {m3['diagonal_sum']}/750 ({m3['diagonal_pct']:.1f}%)\nARI = {m3['ari']:.4f}  |  NMI = {m3['nmi']:.4f}",
        xlabel="Clusters K-Means Baseline (Atributos Escalares, K=3)",
        ylabel="Clusters K-DBA DTW (tslearn, K=3)",
        cmap="Blues",
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.91])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura 34 salva com sucesso: {fig_path}")


def run_kdba_pipeline_for_feature(
    feature: str = "accel_ms2",
    n_windows: int = 750,
    max_iter: int = 15,
    max_iter_barycenter: int = 10,
    random_state: int = 42,
) -> dict:
    """Executa o pipeline completo do K-DBA para a feature especificada."""
    logger.info(f"=== Iniciando Pipeline K-DBA (DTW Barycenter Averaging) para: {feature} ===")
    fcfg = get_feature_paths(feature)
    feat_name = fcfg["name_pt"]

    # 1. Carrega dados brutos (750 janelas, 120s)
    data_matrix, lengths, stems = load_window_series_fast(feature=feature, n_windows=n_windows)
    logger.info(f"Séries temporais carregadas: {data_matrix.shape} (N={n_windows}, L=120s)")

    # 2. Carrega janelas rotuladas existentes
    df_train = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET).copy()
    assert len(df_train) == n_windows, f"Tamanho divergente: df_train={len(df_train)} vs n_windows={n_windows}"

    # Carrega K-Medóides existente se disponível
    kmed_parquet_path = PROCESSED_DATA_DIR / feature / f"dtw_kmedoids_{feature}_clusters_train_750x750.parquet"
    if kmed_parquet_path.exists():
        df_kmed = pd.read_parquet(kmed_parquet_path)
        kmed_labels = df_kmed["kmedoids_dtw_cluster"].to_numpy(dtype=int)
        medoid_for_cluster = {}
        for idx, row in df_kmed[df_kmed["is_medoid_prototype"]].iterrows():
            medoid_for_cluster[int(row["kmedoids_dtw_cluster"])] = int(idx)
        kmedoids_medoids = np.array([medoid_for_cluster[c] for c in range(3)], dtype=int)
    else:
        logger.warning(f"Arquivo K-Medóides não encontrado em {kmed_parquet_path}. Usando zeros.")
        kmed_labels = np.zeros(n_windows, dtype=int)
        kmedoids_medoids = None

    # Carrega Louvain existente
    louv_parquet_path = fcfg["louvain_full_parquet"]
    df_louv = pd.read_parquet(louv_parquet_path).copy()
    louv_raw = df_louv["louvain_community"].values.astype(int)
    louv_sem_labels, _, _ = order_louvain_by_intensity(louv_raw, df_train, feature=feature)

    # Rótulos do K-Means Baseline
    km_labels = df_train["cluster_id"].values.astype(int)

    # 3. Treinamento do K-DBA (TimeSeriesKMeans com metric='dtw')
    logger.info(f"Treinando TimeSeriesKMeans(metric='dtw') com {max_iter} iterações globais e {max_iter_barycenter} iterações DBA...")
    X_3d = data_matrix.reshape((n_windows, 120, 1))

    model = TimeSeriesKMeans(
        n_clusters=3,
        metric="dtw",
        max_iter=max_iter,
        max_iter_barycenter=max_iter_barycenter,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_3d)

    raw_labels = model.labels_.astype(int)
    raw_centers = model.cluster_centers_[:, :, 0]  # Shape: (3, 120)

    # 4. Ordenação e Calibração Física (0: Econômico, 1: Moderado, 2: Agressivo)
    kdba_labels, ordered_centers, raw_to_sem = order_kdba_by_intensity(
        raw_labels, raw_centers, df_train, feature=feature
    )

    logger.info("Distribuição de janelas nos clusters K-DBA ordenados:")
    for c_idx in range(3):
        count = int((kdba_labels == c_idx).sum())
        pct = (count / n_windows) * 100.0
        logger.info(f"  Cluster {c_idx} ({PROFILE_LABELS[c_idx]}): {count} janelas ({pct:.1f}%)")

    # 5. Salva Dataset Processado
    feat_proc_dir = PROCESSED_DATA_DIR / feature
    feat_proc_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "window_idx": np.arange(n_windows),
        "window_file": df_train["window_file"].values,
        "kdba_cluster": kdba_labels,
        "kdba_profile_name": [PROFILE_LABELS[c] for c in kdba_labels],
        "kmedoids_cluster": kmed_labels,
        "louvain_semantic_cluster": louv_sem_labels,
        "kmeans_baseline_cluster": km_labels,
    })

    out_parquet = feat_proc_dir / f"dtw_kdba_{feature}_clusters_train_750x750.parquet"
    out_csv = feat_proc_dir / f"dtw_kdba_{feature}_clusters_train_750x750.csv"
    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)
    logger.info(f"Dataset de clusters K-DBA salvo: {out_parquet}")

    # 6. Constrói Matrizes de Contingência (3x3)
    def build_contingency_df(labels_row: np.ndarray, labels_col: np.ndarray, name_row: str, name_col: str) -> tuple[pd.DataFrame, np.ndarray]:
        mat = np.zeros((3, 3), dtype=int)
        for r, c in zip(labels_row, labels_col):
            mat[int(r), int(c)] += 1

        idx_names = [f"{name_row} C{i}: {PROFILE_SHORT[i]}" for i in range(3)] + ["Total"]
        col_names = [f"{name_col} C{j}: {PROFILE_SHORT[j]}" for j in range(3)] + ["Total"]

        df_ct = pd.DataFrame(index=idx_names, columns=col_names, dtype=int)
        df_ct.iloc[:3, :3] = mat
        df_ct.iloc[:3, 3] = mat.sum(axis=1)
        df_ct.iloc[3, :3] = mat.sum(axis=0)
        df_ct.iloc[3, 3] = int(mat.sum())
        return df_ct, mat

    df_ct_kdba_kmed, mat_kdba_kmed = build_contingency_df(kdba_labels, kmed_labels, "K-DBA", "K-Med")
    df_ct_kdba_louv, mat_kdba_louv = build_contingency_df(kdba_labels, louv_sem_labels, "K-DBA", "Louv")
    df_ct_kdba_km, mat_kdba_km = build_contingency_df(kdba_labels, km_labels, "K-DBA", "K-Means")

    CONTINGENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ct_kdba_kmed.to_csv(CONTINGENCY_RESULTS_DIR / f"tri-04_kdba_vs_kmedoids_{feature}_k3x3.csv")
    df_ct_kdba_louv.to_csv(CONTINGENCY_RESULTS_DIR / f"tri-05_kdba_vs_louvain_{feature}_k3x3.csv")
    df_ct_kdba_km.to_csv(CONTINGENCY_RESULTS_DIR / f"tri-06_kdba_vs_kmeans_{feature}_k3x3.csv")

    # 7. Calcula Métricas de Alinhamento
    metrics_dict = {
        "kdba_vs_kmedoids": compute_alignment_metrics(kdba_labels, kmed_labels, mat_kdba_kmed),
        "kdba_vs_louvain": compute_alignment_metrics(kdba_labels, louv_sem_labels, mat_kdba_louv),
        "kdba_vs_kmeans": compute_alignment_metrics(kdba_labels, km_labels, mat_kdba_km),
    }

    # 8. Estatísticas Físicas por Perfil K-DBA
    profile_rows = []
    for c_idx in range(3):
        mask = kdba_labels == c_idx
        sub = df_train.loc[mask]
        profile_rows.append({
            "cluster_id": c_idx,
            "profile_name": PROFILE_LABELS[c_idx],
            "n_windows": int(mask.sum()),
            "pct_total": round((mask.sum() / n_windows) * 100.0, 2),
            "mean_pos_accel_ms2": round(float(sub["mean_pos_accel_ms2"].mean()), 4),
            "max_pos_accel_ms2": round(float(sub["max_pos_accel_ms2"].mean()), 4),
            "mean_speed_kmh": round(float(sub["mean_speed_kmh"].mean()), 2),
            "max_speed_kmh": round(float(sub["max_speed_kmh"].mean()), 2),
            "mean_power_kw": round(float(sub["mean_power_kw"].mean()), 3),
            "mean_energy_kwh": round(float(sub["total_energy_kwh"].mean()), 4),
            "rapid_accel_rate_min": round(float(sub["rapid_accel_rate_min"].mean()), 3),
        })

    df_profiles = pd.DataFrame(profile_rows)
    feat_res_dir = RESULTS_DIR / feature
    feat_res_dir.mkdir(parents=True, exist_ok=True)
    df_profiles.to_csv(feat_res_dir / f"dtw_kdba_cluster_profiles_{feature}.csv", index=False)

    # 9. Gera Figuras Mestres (300 DPI)
    feat_dir_fig = FIGURES_DIR / feature
    feat_dir_fig.mkdir(parents=True, exist_ok=True)

    fig33_path = feat_dir_fig / f"33_dtw_kdba_barycenters_{feature}.png"
    plot_kdba_barycenters(
        data_matrix=data_matrix,
        lengths=lengths,
        cluster_centers=ordered_centers,
        kdba_labels=kdba_labels,
        df_train=df_train,
        feature=feature,
        fig_path=fig33_path,
        kmedoids_medoids=kmedoids_medoids,
    )

    fig34_path = feat_dir_fig / f"34_triangulation_contingency_kdba_kmedoids_louvain_{feature}.png"
    plot_triangulation_panel(
        df_ct_kdba_kmed=df_ct_kdba_kmed,
        df_ct_kdba_louv=df_ct_kdba_louv,
        df_ct_kdba_km=df_ct_kdba_km,
        metrics_dict=metrics_dict,
        feature=feature,
        fig_path=fig34_path,
    )

    # Copia figuras para o diretório de artefatos
    if ARTIFACT_DIR.exists():
        for src_f in [fig33_path, fig34_path]:
            dst_f = ARTIFACT_DIR / src_f.name
            shutil.copy2(src_f, dst_f)
            logger.info(f"Copiado artefato para brain: {dst_f}")

    # Salva Resumo JSON e CSV
    summary_rows = []
    for pair_key, pair_label in [
        ("kdba_vs_kmedoids", "K-DBA DTW vs. K-Medóides DTW"),
        ("kdba_vs_louvain", "K-DBA DTW vs. Louvain DTW"),
        ("kdba_vs_kmeans", "K-DBA DTW vs. K-Means Euclidiano"),
    ]:
        m = metrics_dict[pair_key]
        summary_rows.append({
            "feature": feature,
            "feature_name_pt": feat_name,
            "comparison_pair": pair_label,
            "diagonal_concordance_count": m["diagonal_sum"],
            "diagonal_concordance_pct": m["diagonal_pct"],
            "adjusted_rand_index": m["ari"],
            "normalized_mutual_info": m["nmi"],
            "rand_index": m["rand_index"],
            "homogeneity": m["homogeneity"],
            "completeness": m["completeness"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(feat_res_dir / f"dtw_kdba_triangulation_summary_{feature}.csv", index=False)

    summary_payload = {
        "feature": feature,
        "n_windows": n_windows,
        "hyperparameters": {
            "n_clusters": 3,
            "metric": "dtw",
            "barycenter_algorithm": "DBA (DTW Barycenter Averaging)",
            "max_iter": max_iter,
            "max_iter_barycenter": max_iter_barycenter,
            "random_state": random_state,
        },
        "cluster_distribution": {
            PROFILE_LABELS[i]: int((kdba_labels == i).sum()) for i in range(3)
        },
        "triangulation_metrics": metrics_dict,
        "cluster_profiles": profile_rows,
    }

    with open(feat_res_dir / f"dtw_kdba_triangulation_summary_{feature}.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Pipeline K-DBA para {feature} finalizado com sucesso! ===")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Agrupamento K-DBA com DTW Barycenter Averaging.")
    parser.add_argument(
        "--feature",
        choices=["accel_ms2", "speed_kmh", "power_pos_kw", "all"],
        default="accel_ms2",
        help="Feature a ser agrupada via K-DBA (padrão: accel_ms2)",
    )
    parser.add_argument("--n_windows", type=int, default=750, help="Número de janelas (padrão: 750)")
    parser.add_argument("--max_iter", type=int, default=10, help="Iterações globais do K-Means (padrão: 10)")
    parser.add_argument("--max_iter_barycenter", type=int, default=5, help="Iterações do DBA (padrão: 5)")
    parser.add_argument("--random_state", type=int, default=42, help="Semente pseudoaleatória (padrão: 42)")

    args = parser.parse_args()

    features_to_run = (
        ["accel_ms2", "power_pos_kw", "speed_kmh"] if args.feature == "all" else [args.feature]
    )

    all_results = {}
    for feat in features_to_run:
        res = run_kdba_pipeline_for_feature(
            feature=feat,
            n_windows=args.n_windows,
            max_iter=args.max_iter,
            max_iter_barycenter=args.max_iter_barycenter,
            random_state=args.random_state,
        )
        all_results[feat] = res

    print("\n" + "=" * 70)
    print("RESUMO EXECUTIVO DO AGRUPAMENTO K-DBA (DTW BARYCENTER AVERAGING)")
    print("=" * 70)
    for feat, res in all_results.items():
        print(f"\n--- Feature: {feat} ---")
        dist = res["cluster_distribution"]
        print(f"Distribuição K=3: Econômico={dist['Econômico / Suave']} | Moderado={dist['Moderado / Regular']} | Agressivo={dist['Agressivo / Dinâmico']}")
        for pair_name, m in res["triangulation_metrics"].items():
            print(f"  {pair_name:20s}: Diagonal={m['diagonal_sum']}/750 ({m['diagonal_pct']:.1f}%) | ARI={m['ari']:.4f} | NMI={m['nmi']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
