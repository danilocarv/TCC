"""
Módulo de Agrupamento K-Medóides com Distância DTW (Time Series Clustering) e Triangulação Metodológica.

Atende à recomendação do orientador Prof. Douglas Braz de aplicar o agrupamento tipo K-Means
diretamente com a métrica DTW (via K-Medóides / PAM - Partitioning Around Medoids), onde o
centro de cada cluster é uma série temporal real (medóide) que minimiza a soma das distâncias
DTW intra-cluster, evitando a distorção de fase da média aritmética ponto a ponto.

Estabelece a triangulação metodológica completa (K = 3 perfis: Econômico, Moderado, Agressivo):
1. K-Means Baseline (Atributos Escalares Agregados + Distância Euclidiana);
2. K-Medóides DTW (Série Temporal 1 Hz Pura + Distância DTW com Medóides Reais);
3. Redes Complexas Louvain DTW (Grafo Completo de Similaridade DTW + Modularidade de Newman-Girvan).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import argparse
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
    silhouette_score,
)

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
    """
    Carrega as séries temporais brutas (1 Hz, 120s) das janelas de treino sem depender do Numba.
    """
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
            vals = np.maximum(0.0, vals)
        length = min(len(vals), max_len)
        data_matrix[idx, :length] = vals[:length]
        lengths[idx] = length
        stems.append(fpath.stem)

    return data_matrix, lengths, stems


def fit_pam_kmedoids(
    dist_matrix: np.ndarray,
    k: int = 3,
    n_init: int = 25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Executa o algoritmo PAM (Partitioning Around Medoids) exato sobre a matriz de distâncias DTW:
    1. Testa a inicialização determinística BUILD (Kaufman & Rousseeuw) + fase SWAP global;
    2. Testa `n_init` inicializações K-Medoids++ com refinamento Lloyd-PAM + fase SWAP global;
    3. Retorna a partição de menor custo (soma total das distâncias DTW de cada ponto ao seu medóide).
    """
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(random_state)

    def _pam_swap_optimize(init_medoids: np.ndarray) -> tuple[np.ndarray, float, int]:
        medoids = init_medoids.copy()
        iters = 0
        # Primeiro aplica atualizações rápidas de centroide-medóide (estilo Lloyd)
        for _ in range(50):
            iters += 1
            labels = np.argmin(dist_matrix[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) > 0:
                    sub_d = dist_matrix[np.ix_(members, members)]
                    new_medoids[c] = members[int(np.argmin(sub_d.sum(axis=1)))]
            if np.array_equal(medoids, new_medoids):
                break
            medoids = new_medoids

        # Em seguida, aplica a fase SWAP exata de Kaufman & Rousseeuw (testa todos os k * n pares)
        while True:
            iters += 1
            best_cost = float(np.sum(np.min(dist_matrix[:, medoids], axis=1)))
            best_swap = None
            for idx_m in range(k):
                other_medoids = np.delete(medoids, idx_m)
                dist_other = np.min(dist_matrix[:, other_medoids], axis=1)
                candidate_costs = np.minimum(dist_other[:, None], dist_matrix).sum(axis=0)
                # Proíbe escolher um medóide já ativo
                candidate_costs[other_medoids] = np.inf
                min_h = int(np.argmin(candidate_costs))
                if candidate_costs[min_h] < best_cost - 1e-9:
                    best_cost = float(candidate_costs[min_h])
                    best_swap = (idx_m, min_h)
            if best_swap is None:
                break
            medoids[best_swap[0]] = best_swap[1]

        final_cost = float(np.sum(np.min(dist_matrix[:, medoids], axis=1)))
        return medoids, final_cost, iters

    # 1. Inicialização determinística BUILD
    m0 = int(np.argmin(dist_matrix.sum(axis=1)))
    build_medoids = [m0]
    for _ in range(1, k):
        dist_nearest = np.min(dist_matrix[:, build_medoids], axis=1)
        gains = np.maximum(0.0, dist_nearest[:, None] - dist_matrix).sum(axis=0)
        gains[build_medoids] = -1.0
        build_medoids.append(int(np.argmax(gains)))

    best_medoids, best_cost, best_iters = _pam_swap_optimize(np.array(build_medoids, dtype=int))

    # 2. Múltiplas inicializações K-Medoids++ para garantir ótimo global
    for _ in range(n_init):
        m_init = [int(rng.integers(0, n))]
        for _ in range(1, k):
            d_near = np.min(dist_matrix[:, m_init], axis=1)
            probs = d_near ** 2
            total_p = probs.sum()
            if total_p > 0:
                probs = probs / total_p
                m_init.append(int(rng.choice(n, p=probs)))
            else:
                cand = [ idx for idx in range(n) if idx not in m_init ]
                m_init.append(int(rng.choice(cand)))
        cand_medoids, cand_cost, cand_iters = _pam_swap_optimize(np.array(m_init, dtype=int))
        if cand_cost < best_cost - 1e-9:
            best_medoids = cand_medoids
            best_cost = cand_cost
            best_iters = cand_iters

    final_labels = np.argmin(dist_matrix[:, best_medoids], axis=1)
    return final_labels, best_medoids, best_cost, best_iters


def order_clusters_by_intensity(
    labels: np.ndarray,
    medoids: np.ndarray,
    df_train: pd.DataFrame,
    feature: str = "accel_ms2",
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """
    Ordena os K=3 clusters de forma crescente de intensidade dinâmica:
    - Para 'accel_ms2': ordena por 'mean_pos_accel_ms2' médio do cluster;
    - Para 'speed_kmh': ordena por 'mean_speed_kmh' médio do cluster;
    - Para 'power_pos_kw': ordena por 'mean_power_kw' médio do cluster.
    Retorna (ordered_labels, ordered_medoids, mapping_old_to_new).
    """
    sort_col_map = {
        "accel_ms2": "mean_pos_accel_ms2",
        "speed_kmh": "mean_speed_kmh",
        "power_pos_kw": "mean_power_kw",
    }
    sort_col = sort_col_map.get(feature, "mean_pos_accel_ms2")

    k = len(np.unique(labels))
    cluster_means = []
    for c in range(k):
        mask = labels == c
        val = float(df_train.loc[mask, sort_col].mean()) if mask.any() else 0.0
        cluster_means.append((c, val))

    # Ordena do menor (0: Econômico) para o maior (2: Agressivo)
    cluster_means.sort(key=lambda x: x[1])
    old_to_new = {old_c: new_c for new_c, (old_c, _) in enumerate(cluster_means)}

    ordered_labels = np.array([old_to_new[int(c)] for c in labels], dtype=int)
    ordered_medoids = np.zeros_like(medoids)
    for old_c, new_c in old_to_new.items():
        ordered_medoids[new_c] = medoids[old_c]

    return ordered_labels, ordered_medoids, old_to_new


def order_louvain_by_intensity(
    louv_raw: np.ndarray,
    df_train: pd.DataFrame,
    feature: str = "accel_ms2",
) -> tuple[np.ndarray, dict[int, int], dict[int, str]]:
    """
    Mapeia as comunidades Louvain (C0, C1, C2...) para a ordem física de intensidade
    (Econômico -> Moderado -> Agressivo), preservando o ID original no rótulo textual
    (ex: 'C1: Econ. (N=271)').
    """
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


def plot_medoid_prototypes(
    data_matrix: np.ndarray,
    lengths: np.ndarray,
    ordered_medoids: np.ndarray,
    kmed_labels: np.ndarray,
    df_train: pd.DataFrame,
    dist_matrix: np.ndarray,
    feature: str,
    fig_path: Path,
) -> None:
    """
    Gera a Figura 30: Painel das 3 Séries Temporais Reais Eleitas como Medóides Protótipos
    (120 segundos, 1 Hz) + Comparativo Físico dos 3 Grupos K-Medóides DTW.
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
        f"K-Medóides com Distância DTW ({feat_name}) — Séries Temporais Reais dos 3 Medóides Protótipos (120s)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.938,
        "Cada curva abaixo é a janela real de 120s eleita pelo algoritmo PAM como o centro físico (menor soma de distâncias DTW) do seu cluster",
        ha="center",
        fontsize=10.5,
        color="#475569",
        style="italic",
    )

    # Limites globais do eixo Y para comparação visual justa na mesma escala
    all_medoid_vals = np.concatenate([data_matrix[m, :lengths[m]] for m in ordered_medoids])
    y_min = min(float(np.min(all_medoid_vals)) * 1.15, -0.5 if feature == "accel_ms2" else 0.0)
    y_max = float(np.max(all_medoid_vals)) * 1.18

    for c_idx in range(3):
        ax = fig.add_subplot(gs[c_idx, :])
        m_idx = int(ordered_medoids[c_idx])
        seq_len = int(lengths[m_idx])
        series = data_matrix[m_idx, :seq_len]
        t = np.arange(seq_len)

        row_meta = df_train.iloc[m_idx]
        cluster_mask = kmed_labels == c_idx
        n_members = int(cluster_mask.sum())
        pct_members = (n_members / len(kmed_labels)) * 100.0

        # Estatísticas médias do cluster inteiro
        cl_mean_accel = float(df_train.loc[cluster_mask, "mean_pos_accel_ms2"].mean())
        cl_max_accel = float(df_train.loc[cluster_mask, "max_pos_accel_ms2"].mean())
        cl_mean_speed = float(df_train.loc[cluster_mask, "mean_speed_kmh"].mean())
        cl_rapid_rate = float(df_train.loc[cluster_mask, "rapid_accel_rate_min"].mean())

        color = PROFILE_COLORS[c_idx]
        prof_title = PROFILE_LABELS[c_idx]

        # Preenchimento sob a curva
        if feature == "accel_ms2":
            ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.8)
            ax.fill_between(t, 0, series, where=(series >= 0), color=color, alpha=0.22, interpolate=True, label="Aceleração (+)")
            ax.fill_between(t, 0, series, where=(series < 0), color="#64748B", alpha=0.18, interpolate=True, label="Frenagem (-)")
        else:
            ax.fill_between(t, 0, series, color=color, alpha=0.22)

        ax.plot(t, series, color=color, linewidth=2.0, label=f"Medóide #{m_idx} ({row_meta['window_file']})")

        # Destaca picos extremos na série do medóide
        max_t = int(np.argmax(series))
        min_t = int(np.argmin(series))
        y_span = max(y_max - y_min, 1.0)
        ax.scatter([max_t], [series[max_t]], color="#1E293B", s=35, zorder=5)
        text_x_peak = (max_t - 15) if max_t > 72 else min(max_t + 2, 96)
        ax.annotate(
            f"Pico: {series[max_t]:.2f} {unit}" if feature != "accel_ms2" else f"Pico: {series[max_t]:+.2f} {unit}",
            xy=(max_t, series[max_t]),
            xytext=(text_x_peak, series[max_t] + y_span * 0.04),
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )
        if feature == "accel_ms2" and series[min_t] < -0.5:
            ax.scatter([min_t], [series[min_t]], color="#475569", s=30, zorder=5)
            ax.annotate(
                f"Fren.: {series[min_t]:+.2f} {unit}",
                xy=(min_t, series[min_t]),
                xytext=(min(min_t + 2, 96), series[min_t] - y_span * 0.07),
                fontsize=8.0,
                color="#475569",
            )

        ax.set_xlim(0, 120)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")
        ax.set_ylabel(f"{feat_name}\n({unit})", fontsize=9.5, fontweight="bold")
        if c_idx == 2:
            ax.set_xlabel("Tempo na Janela de Condução Ativa (segundos)", fontsize=10, fontweight="bold")

        cl_mean_power = float(df_train.loc[cluster_mask, "mean_power_kw"].mean())
        title_str = (
            f"Cluster {c_idx} — {prof_title}  |  N = {n_members} janelas ({pct_members:.1f}%)  |  "
            f"Medóide Real: #{m_idx} (Veículo {int(row_meta['VehId'])}, Viagem {int(row_meta['Trip'])}, Janela {int(row_meta['window_id'])})"
        )
        subtitle_stats = (
            f"Médias do Grupo: Veloc. = {cl_mean_speed:.1f} km/h  |  Acel. Positiva = {cl_mean_accel:.3f} m/s²  |  "
            f"Pico Acel. = {cl_max_accel:.2f} m/s²  |  Potência Média = {cl_mean_power:.2f} kW"
        )
        ax.set_title(f"{title_str}\n{subtitle_stats}", fontsize=9.8, fontweight="bold", color="#1E293B", loc="left", pad=6)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # Linha 4: 3 gráficos de barras comparando a progressão física dos 3 clusters K-Medóides
    if feature == "speed_kmh":
        metrics_to_plot = [
            ("mean_speed_kmh", "Velocidade Média do Cluster (km/h)", "km/h"),
            ("max_speed_kmh", "Velocidade Máxima Média (km/h)", "km/h"),
            ("mean_pos_accel_ms2", "Aceleração Positiva Média (m/s²)", "m/s²"),
        ]
    elif feature == "power_pos_kw":
        metrics_to_plot = [
            ("mean_power_kw", "Potência Elétrica Média (kW)", "kW"),
            ("mean_pos_accel_ms2", "Aceleração Positiva Média (m/s²)", "m/s²"),
            ("mean_speed_kmh", "Velocidade Média (km/h)", "km/h"),
        ]
    else:
        metrics_to_plot = [
            ("mean_pos_accel_ms2", "Aceleração Positiva Média (m/s²)", "m/s²"),
            ("max_pos_accel_ms2", "Pico Médio de Aceleração (m/s²)", "m/s²"),
            ("rapid_accel_rate_min", "Taxa de Arrancadas Bruscas (/min)", "ev/min"),
        ]

    for m_i, (col_m, m_title, m_unit) in enumerate(metrics_to_plot):
        ax_b = fig.add_subplot(gs[3, m_i])
        means = [float(df_train.loc[kmed_labels == c, col_m].mean()) for c in range(3)]
        x_pos = np.arange(3)
        bars = ax_b.bar(
            x_pos,
            means,
            color=[PROFILE_COLORS[0], PROFILE_COLORS[1], PROFILE_COLORS[2]],
            width=0.55,
            edgecolor="#1E293B",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar, val in zip(bars, means):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(means) * 0.03,
                f"{val:.3f}" if val < 1.0 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#0F172A",
            )
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([f"C{c}: {PROFILE_SHORT[c]}" for c in range(3)], fontsize=9.5, fontweight="bold")
        ax_b.set_title(m_title, fontsize=10, fontweight="bold", pad=8)
        ax_b.set_ylabel(m_unit, fontsize=9)
        ax_b.set_ylim(0, max(means) * 1.24 if max(means) > 0 else 1.0)
        ax_b.grid(axis="y", linestyle=":", alpha=0.5)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Figura 30 salva em: {fig_path}")

    if ARTIFACT_DIR.exists():
        art_target = ARTIFACT_DIR / fig_path.name
        shutil.copy2(fig_path, art_target)
        logger.info(f"Copiada para artefato: {art_target}")


def plot_single_3x3_heatmap(
    ax: plt.Axes,
    df_ct: pd.DataFrame,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "Blues",
) -> None:
    """
    Plota uma matriz de contingência 3x3 com contagens inteiras reais e porcentagens por linha.
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
        cbar_kws={"label": "Nº de Janelas", "shrink": 0.78, "pad": 0.03},
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=10.2, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel(xlabel, fontsize=9.8, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=9.8, fontweight="bold", labelpad=6)
    ax.tick_params(axis="x", labelsize=8.8, rotation=0)
    ax.tick_params(axis="y", labelsize=8.8, rotation=0)


def plot_triangulation_panel(
    df_ct_kmed_louv: pd.DataFrame,
    df_ct_kmed_km: pd.DataFrame,
    df_ct_louv_km: pd.DataFrame,
    metrics_dict: dict,
    feature: str,
    fig_path: Path,
) -> None:
    """
    Gera a Figura 31: Painel Triplo de Contingência da Triangulação Metodológica (300 DPI).
    """
    fcfg = get_feature_paths(feature)
    feat_name = fcfg["name_pt"]

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.2), dpi=300)
    fig.subplots_adjust(wspace=0.62, top=0.77, bottom=0.16)

    fig.suptitle(
        f"Triangulação Metodológica ({feat_name}, K=3) — K-Medóides DTW vs. Louvain DTW vs. K-Means Euclidiano",
        fontsize=14.5,
        fontweight="bold",
        y=0.96,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.90,
        "Comparação em contagens inteiras reais (N = 750 janelas de treino de 120s) e percentual por linha, ordenadas por intensidade física (Econômico → Moderado → Agressivo)",
        ha="center",
        fontsize=10.5,
        color="#475569",
        style="italic",
    )

    # Painel 1: K-Medóides DTW vs. Louvain DTW
    m1 = metrics_dict["kmedoids_vs_louvain"]
    plot_single_3x3_heatmap(
        ax=axes[0],
        df_ct=df_ct_kmed_louv,
        title="1. K-Medóides DTW vs. Louvain DTW (Série 1 Hz)",
        subtitle=f"Diagonal: {m1['diagonal_sum']}/750 ({m1['diagonal_pct']:.1f}%)\nARI = {m1['ari']:.4f}  |  NMI = {m1['nmi']:.4f}",
        xlabel="Comunidades Louvain DTW (Grafo Completo, K=3)",
        ylabel="Clusters K-Medóides DTW (PAM, K=3)",
        cmap="Greens",
    )

    # Painel 2: K-Medóides DTW vs. K-Means Baseline
    m2 = metrics_dict["kmedoids_vs_kmeans"]
    plot_single_3x3_heatmap(
        ax=axes[1],
        df_ct=df_ct_kmed_km,
        title="2. K-Medóides DTW vs. K-Means Baseline (Tabular)",
        subtitle=f"Diagonal: {m2['diagonal_sum']}/750 ({m2['diagonal_pct']:.1f}%)\nARI = {m2['ari']:.4f}  |  NMI = {m2['nmi']:.4f}",
        xlabel="Clusters K-Means Baseline (Atributos Escalares, K=3)",
        ylabel="Clusters K-Medóides DTW (PAM, K=3)",
        cmap="Blues",
    )

    # Painel 3: Louvain DTW vs. K-Means Baseline
    m3 = metrics_dict["louvain_vs_kmeans"]
    plot_single_3x3_heatmap(
        ax=axes[2],
        df_ct=df_ct_louv_km,
        title="3. Louvain DTW vs. K-Means Baseline (Tabular)",
        subtitle=f"Diagonal: {m3['diagonal_sum']}/750 ({m3['diagonal_pct']:.1f}%)\nARI = {m3['ari']:.4f}  |  NMI = {m3['nmi']:.4f}",
        xlabel="Clusters K-Means Baseline (Atributos Escalares, K=3)",
        ylabel="Comunidades Louvain DTW (Grafo Completo, K=3)",
        cmap="Oranges",
    )

    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Figura 31 salva em: {fig_path}")

    if ARTIFACT_DIR.exists():
        art_target = ARTIFACT_DIR / fig_path.name
        shutil.copy2(fig_path, art_target)
        logger.info(f"Copiada para artefato: {art_target}")


def compute_alignment_metrics(labels_a: np.ndarray, labels_b: np.ndarray, mat_3x3: np.ndarray) -> dict:
    """
    Calcula métricas formais de concordância entre duas partições K=3.
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


def run_dtw_kmedoids_pipeline(feature: str = "accel_ms2") -> dict:
    """
    Executa o pipeline completo de K-Medóides DTW (K=3) e a triangulação metodológica para a feature informada.
    """
    ensure_directories()
    CONTINGENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fcfg = get_feature_paths(feature)
    feat_dir_proc = PROCESSED_DATA_DIR / feature
    feat_dir_fig = FIGURES_DIR / feature
    feat_dir_res = RESULTS_DIR / feature
    feat_dir_proc.mkdir(parents=True, exist_ok=True)
    feat_dir_fig.mkdir(parents=True, exist_ok=True)
    feat_dir_res.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 95)
    logger.info(f"INICIANDO PIPELINE K-MEDÓIDES DTW (PAM, K=3) — FEATURE: {feature} ({fcfg['name_pt']})")
    logger.info("=" * 95)

    # 1. Carrega Matriz de Distâncias DTW pré-computada (750x750)
    dtw_npy_path = fcfg["dtw_full_npy"]
    if not dtw_npy_path.exists():
        raise FileNotFoundError(f"Matriz DTW não encontrada em: {dtw_npy_path}")

    dist_matrix = np.load(dtw_npy_path)
    np.fill_diagonal(dist_matrix, 0.0)
    n_windows = dist_matrix.shape[0]
    logger.info(f"Matriz DTW carregada: {dtw_npy_path} | Shape: {dist_matrix.shape}")

    # 2. Carrega janelas rotuladas pelo K-Means Baseline e Comunidades Louvain DTW
    df_train = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET).copy()
    louv_parquet_path = fcfg["louvain_full_parquet"]
    df_louv = pd.read_parquet(louv_parquet_path).copy()

    assert len(df_train) == n_windows, f"Divergência de tamanho: df_train={len(df_train)} vs DTW={n_windows}"
    assert (df_train["window_file"].values == df_louv["window_file"].values).all(), (
        "Alinhamento de janelas entre K-Means e Louvain não é estritamente idêntico!"
    )

    # 3. Ajusta K-Medóides (PAM) exato com K=3
    raw_kmed_labels, raw_medoids, total_dtw_cost, n_iters = fit_pam_kmedoids(
        dist_matrix, k=3, n_init=25, random_state=42
    )
    kmed_labels, ordered_medoids, _ = order_clusters_by_intensity(
        raw_kmed_labels, raw_medoids, df_train, feature=feature
    )
    sil_dtw = float(silhouette_score(dist_matrix, kmed_labels, metric="precomputed"))
    logger.info(
        f"PAM K-Medóides convergiu em {n_iters} passos | Custo Total DTW = {total_dtw_cost:.2f} | "
        f"Silhouette DTW = {sil_dtw:.4f} | Medóides ordenados = {ordered_medoids.tolist()}"
    )

    # 4. Prepara rótulos do K-Means Baseline (0: Econ, 1: Mod, 2: Agres) e Louvain DTW ordenado por intensidade
    km_labels = df_train["cluster_id"].values.astype(int)
    louv_raw = df_louv["louvain_community"].values.astype(int)
    louv_sem_labels, louv_orig_to_sem, louv_axis_names = order_louvain_by_intensity(
        louv_raw, df_train, feature=feature
    )

    # 5. Salva dataset processado com os clusters do K-Medóides (sem alterar arquivos existentes)
    df_out = df_louv.copy()
    df_out["kmedoids_dtw_cluster"] = kmed_labels
    df_out["kmedoids_dtw_label"] = [PROFILE_LABELS[int(c)] for c in kmed_labels]
    df_out["is_medoid_prototype"] = False
    df_out.loc[ordered_medoids, "is_medoid_prototype"] = True
    df_out["dist_to_cluster_medoid_dtw"] = [
        float(dist_matrix[i, ordered_medoids[kmed_labels[i]]]) for i in range(n_windows)
    ]
    df_out["louvain_semantic_cluster"] = louv_sem_labels
    df_out["kmeans_baseline_cluster"] = km_labels

    out_parquet = feat_dir_proc / f"dtw_kmedoids_{feature}_clusters_train_750x750.parquet"
    out_csv = feat_dir_proc / f"dtw_kmedoids_{feature}_clusters_train_750x750.csv"
    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Clusters K-Medóides DTW salvos em: {out_parquet}")

    # 6. Constrói as 3 Matrizes de Contingência 3x3 da Triangulação
    kmed_counts = pd.Series(kmed_labels).value_counts().to_dict()
    km_counts = pd.Series(km_labels).value_counts().to_dict()

    kmed_axis_names = {
        c: f"KMed {c}: {PROFILE_SHORT[c]}\n(Med #{ordered_medoids[c]}, N={kmed_counts.get(c, 0)})"
        for c in range(3)
    }
    km_axis_names = {
        c: f"KM {c}: {PROFILE_SHORT[c]}\n(N={km_counts.get(c, 0)})"
        for c in range(3)
    }

    mat_kmed_louv = np.zeros((3, 3), dtype=int)
    mat_kmed_km = np.zeros((3, 3), dtype=int)
    mat_louv_km = np.zeros((3, 3), dtype=int)

    for i in range(n_windows):
        mat_kmed_louv[kmed_labels[i], louv_sem_labels[i]] += 1
        mat_kmed_km[kmed_labels[i], km_labels[i]] += 1
        mat_louv_km[louv_sem_labels[i], km_labels[i]] += 1

    df_ct_kmed_louv = pd.DataFrame(
        mat_kmed_louv,
        index=[kmed_axis_names[c] for c in range(3)],
        columns=[louv_axis_names[c] for c in range(3)],
    )
    df_ct_kmed_km = pd.DataFrame(
        mat_kmed_km,
        index=[kmed_axis_names[c] for c in range(3)],
        columns=[km_axis_names[c] for c in range(3)],
    )
    df_ct_louv_km = pd.DataFrame(
        mat_louv_km,
        index=[louv_axis_names[c] for c in range(3)],
        columns=[km_axis_names[c] for c in range(3)],
    )

    # Salva CSVs de contingência com margens
    for code_prefix, df_ct in [
        (f"tri-01_kmedoids_vs_louvain_{feature}_k3x3", df_ct_kmed_louv),
        (f"tri-02_kmedoids_vs_kmeans_{feature}_k3x3", df_ct_kmed_km),
        (f"tri-03_louvain_vs_kmeans_{feature}_k3x3", df_ct_louv_km),
    ]:
        df_save = df_ct.copy()
        df_save.index = [idx.replace("\n", " ") for idx in df_save.index]
        df_save.columns = [col.replace("\n", " ") for col in df_save.columns]
        df_save["Total_Linha"] = df_save.sum(axis=1)
        df_save.loc["Total_Coluna"] = df_save.sum(axis=0)
        df_save.to_csv(CONTINGENCY_RESULTS_DIR / f"{code_prefix}.csv", encoding="utf-8-sig")

    # 7. Calcula métricas de alinhamento e estatísticas físicas por grupo
    metrics_dict = {
        "kmedoids_vs_louvain": compute_alignment_metrics(kmed_labels, louv_sem_labels, mat_kmed_louv),
        "kmedoids_vs_kmeans": compute_alignment_metrics(kmed_labels, km_labels, mat_kmed_km),
        "louvain_vs_kmeans": compute_alignment_metrics(louv_sem_labels, km_labels, mat_louv_km),
    }

    medoids_info = []
    profiles_rows = []
    for c in range(3):
        m_idx = int(ordered_medoids[c])
        r_meta = df_train.iloc[m_idx]
        mask_kmed = kmed_labels == c
        mask_louv = louv_sem_labels == c
        mask_km = km_labels == c

        medoids_info.append({
            "cluster_id": c,
            "profile_label": PROFILE_LABELS[c],
            "medoid_window_index": m_idx,
            "window_file": str(r_meta["window_file"]),
            "VehId": int(r_meta["VehId"]),
            "Trip": int(r_meta["Trip"]),
            "window_id": int(r_meta["window_id"]),
            "cluster_size": int(mask_kmed.sum()),
            "cluster_share_pct": round(float(mask_kmed.mean() * 100.0), 2),
            "medoid_mean_pos_accel_ms2": round(float(r_meta["mean_pos_accel_ms2"]), 4),
            "medoid_max_pos_accel_ms2": round(float(r_meta["max_pos_accel_ms2"]), 4),
            "medoid_mean_speed_kmh": round(float(r_meta["mean_speed_kmh"]), 2),
            "cluster_mean_pos_accel_ms2": round(float(df_train.loc[mask_kmed, "mean_pos_accel_ms2"].mean()), 4),
            "cluster_max_pos_accel_ms2": round(float(df_train.loc[mask_kmed, "max_pos_accel_ms2"].mean()), 4),
            "cluster_rapid_accel_rate_min": round(float(df_train.loc[mask_kmed, "rapid_accel_rate_min"].mean()), 4),
            "cluster_mean_speed_kmh": round(float(df_train.loc[mask_kmed, "mean_speed_kmh"].mean()), 2),
        })

        for method_name, m_mask in [
            ("K-Medoids DTW", mask_kmed),
            ("Louvain DTW", mask_louv),
            ("K-Means Euclidiano", mask_km),
        ]:
            profiles_rows.append({
                "feature": feature,
                "method": method_name,
                "cluster_order": c,
                "profile_label": PROFILE_LABELS[c],
                "n_windows": int(m_mask.sum()),
                "share_pct": round(float(m_mask.mean() * 100.0), 2),
                "mean_pos_accel_ms2": round(float(df_train.loc[m_mask, "mean_pos_accel_ms2"].mean()), 4),
                "max_pos_accel_ms2": round(float(df_train.loc[m_mask, "max_pos_accel_ms2"].mean()), 4),
                "rapid_accel_rate_min": round(float(df_train.loc[m_mask, "rapid_accel_rate_min"].mean()), 4),
                "hard_braking_rate_min": round(float(df_train.loc[m_mask, "hard_braking_rate_min"].mean()), 4),
                "mean_speed_kmh": round(float(df_train.loc[m_mask, "mean_speed_kmh"].mean()), 2),
                "mean_power_kw": round(float(df_train.loc[m_mask, "mean_power_kw"].mean()), 3),
            })

    df_profiles = pd.DataFrame(profiles_rows)
    profiles_csv_path = feat_dir_res / f"dtw_kmedoids_cluster_profiles_{feature}.csv"
    df_profiles.to_csv(profiles_csv_path, index=False, encoding="utf-8-sig")

    summary_payload = {
        "feature": feature,
        "feature_name_pt": fcfg["name_pt"],
        "n_windows": n_windows,
        "k_clusters": 3,
        "pam_optimization": {
            "total_dtw_inertia": round(total_dtw_cost, 4),
            "mean_dtw_distance_to_medoid": round(total_dtw_cost / n_windows, 4),
            "silhouette_precomputed_dtw": round(sil_dtw, 4),
            "iterations": n_iters,
        },
        "medoids_prototypes": medoids_info,
        "triangulation_alignment": metrics_dict,
    }

    summary_json_path = feat_dir_res / f"dtw_kmedoids_triangulation_summary_{feature}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    summary_rows = []
    for pair_key, pair_label in [
        ("kmedoids_vs_louvain", "K-Medóides DTW vs. Louvain DTW"),
        ("kmedoids_vs_kmeans", "K-Medóides DTW vs. K-Means Euclidiano"),
        ("louvain_vs_kmeans", "Louvain DTW vs. K-Means Euclidiano"),
    ]:
        m = metrics_dict[pair_key]
        summary_rows.append({
            "feature": feature,
            "comparison": pair_label,
            **m,
        })
    df_sum_csv = pd.DataFrame(summary_rows)
    df_sum_csv.to_csv(feat_dir_res / f"dtw_kmedoids_triangulation_summary_{feature}.csv", index=False, encoding="utf-8-sig")

    # 8. Carrega séries temporais brutas e plota Figuras 30 e 31
    data_matrix, lengths, _ = load_window_series_fast(feature=feature, n_windows=n_windows)

    fig30_path = feat_dir_fig / f"30_dtw_kmedoids_prototypes_{feature}.png"
    plot_medoid_prototypes(
        data_matrix=data_matrix,
        lengths=lengths,
        ordered_medoids=ordered_medoids,
        kmed_labels=kmed_labels,
        df_train=df_train,
        dist_matrix=dist_matrix,
        feature=feature,
        fig_path=fig30_path,
    )

    fig31_path = feat_dir_fig / f"31_triangulation_contingency_kmedoids_louvain_kmeans_{feature}.png"
    plot_triangulation_panel(
        df_ct_kmed_louv=df_ct_kmed_louv,
        df_ct_kmed_km=df_ct_kmed_km,
        df_ct_louv_km=df_ct_louv_km,
        metrics_dict=metrics_dict,
        feature=feature,
        fig_path=fig31_path,
    )

    logger.info("Resumo da Triangulação Metodológica:")
    logger.info(f"\n{df_sum_csv.to_string(index=False)}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Agrupamento K-Medóides com Distância DTW e Triangulação.")
    parser.add_argument(
        "--feature",
        type=str,
        default="accel_ms2",
        choices=["accel_ms2", "speed_kmh", "power_pos_kw", "all"],
        help="Variável temporal para agrupamento K-Medóides DTW (padrão: accel_ms2).",
    )
    args = parser.parse_args()

    if args.feature == "all":
        for feat in ["accel_ms2", "speed_kmh", "power_pos_kw"]:
            run_dtw_kmedoids_pipeline(feat)
    else:
        run_dtw_kmedoids_pipeline(args.feature)


if __name__ == "__main__":
    main()
