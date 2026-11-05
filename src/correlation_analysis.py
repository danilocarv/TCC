"""
Pipeline Completo de Análise de Similaridade por Correlação de Pearson e Redes Complexas.

Executa a cadeia metodológica completa de 4 cenários topológicos sobre a matriz de correlação:
1. Cenário 1: Grafo Completo de Correlações Positivas (r > 0) + Louvain Puro;
2. Cenário 2: Árvore Geradora Mínima (MST de Kruskal com Métrica de Mantegna) + Louvain Puro;
3. Cenário 3A: Filtragem por Percentil 25 Direto (Top 25% correlações mais fortes) + Louvain Puro;
4. Cenário 3B: Rede Otimizada (União MST ∪ P25) + Louvain Puro;
5. Síntese Comparativa dos 4 Cenários e Cruzamento com DTW e K-Means (ARI e NMI).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import (
    WINDOWS_TRAIN_PARQUET_DIR,
    PROCESSED_FEATURES_DIR,
    DTW_PILOT_SIZE,
    LOUVAIN_RANDOM_SEED,
    FEATURE_REGISTRY,
    get_feature_paths,
    CORRELATION_SIMILARITY_METHOD,
)


def load_and_resample_series(
    feature: str = "speed_kmh",
    n_windows: int | None = None,
    n_points: int = 100,
) -> tuple[np.ndarray, list[str]]:
    """
    Carrega as séries temporais de treino da feature especificada e as reamostra
    (via interpolação linear) para um grid padronizado de n_points (0% a 100% de progresso).
    
    Retorna a matriz X de dimensão (K, n_points) e a lista com os nomes dos arquivos.
    """
    fcfg = get_feature_paths(feature)
    col_name = fcfg["column"]

    train_files = sorted(WINDOWS_TRAIN_PARQUET_DIR.glob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(
            f"Nenhum arquivo Parquet encontrado em {WINDOWS_TRAIN_PARQUET_DIR}!"
        )

    if n_windows is not None:
        train_files = train_files[:n_windows]

    k = len(train_files)
    grid_target = np.linspace(0.0, 1.0, n_points)
    X = np.zeros((k, n_points), dtype=np.float64)
    file_stems = []

    clip_positive = fcfg.get("clip_positive", False)

    for idx, fpath in enumerate(train_files):
        df_win = pd.read_parquet(fpath, columns=[col_name])
        vals = df_win[col_name].to_numpy(dtype=np.float64)

        if clip_positive:
            vals = np.maximum(0.0, vals)

        n_samples = len(vals)
        if n_samples < 2:
            X[idx, :] = vals[0] if n_samples == 1 else 0.0
        else:
            orig_grid = np.linspace(0.0, 1.0, n_samples)
            X[idx, :] = np.interp(grid_target, orig_grid, vals)

        file_stems.append(fpath.stem)

    return X, file_stems


def compute_pearson_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcula de forma vetorizada a matriz simétrica de correlação de Pearson R (K x K).
    R_ij = Cov(X_i, X_j) / (sigma_i * sigma_j).
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)

    valid_std = stds > 1e-8
    stds_safe = np.where(valid_std, stds, 1.0)

    Z = (X - means) / stds_safe
    Z = np.where(valid_std, Z, 0.0)

    m = X.shape[1]
    corr = (Z @ Z.T) / float(m)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return corr


# ==============================================================================
# CONVERSÃO DE CORRELAÇÃO EM MÉTRICA DE SIMILARIDADE NÃO-NEGATIVA [0, 1]
# ==============================================================================

def correlation_to_similarity(
    corr_matrix: np.ndarray,
    method: str = CORRELATION_SIMILARITY_METHOD,
) -> np.ndarray:
    """
    Converte a matriz de correlação de Pearson R in [-1, 1] em uma métrica de similaridade estrita S in [0, 1].
    Garante que correlações negativas não são descartadas, mas sim mapeadas em [0, 0.5) de forma suave.
    
    Métodos:
    - 'linear_normalized': S = (1 + r) / 2 = 1 - |1 - r| / 2 (Interpretação exata de '1 - módulo de N').
      r = +1.0 -> S = 1.0; r = 0.0 -> S = 0.5; r = -1.0 -> S = 0.0.
    - 'mantegna_normalized': S = 1 - sqrt(2*(1 - r)) / 2.
      r = +1.0 -> S = 1.0; r = 0.0 -> S = 0.293; r = -1.0 -> S = 0.0.
    - 'exponential_kernel': S = exp(-sqrt(2*(1 - r)) / sigma) onde sigma = mediana(d_Mantegna).
    """
    if method == "linear_normalized":
        # S = 1 - |1 - r| / 2 = (1 + r) / 2
        sim = (1.0 + corr_matrix) / 2.0
    elif method == "mantegna_normalized":
        d_mantegna = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr_matrix)))
        sim = 1.0 - (d_mantegna / 2.0)
    elif method == "exponential_kernel":
        d_mantegna = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr_matrix)))
        off_diag = d_mantegna[np.triu_indices_from(d_mantegna, k=1)]
        sigma = float(np.median(off_diag))
        if sigma <= 0.0:
            sigma = 1.0
        sim = np.exp(-d_mantegna / sigma)
    else:
        raise ValueError(f"Método de similaridade desconhecido: {method}")

    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


# ==============================================================================
# CONSTRUÇÃO DE GRAFOS: CENÁRIOS 1, 2, 3A e 3B
# ==============================================================================

def build_full_correlation_graph(
    corr_matrix: np.ndarray,
    sim_matrix: np.ndarray | None = None,
) -> tuple[nx.Graph, dict]:
    """
    Cenário 1: Constrói o grafo completo não direcionado ponderado pela métrica de similaridade
    estrita S_ij in [0, 1] (onde r in [-1, 1] é preservado continuamente sem descarte de negativos).
    W_ij = S_ij para i != j.
    """
    if sim_matrix is None:
        sim_matrix = correlation_to_similarity(corr_matrix, method=CORRELATION_SIMILARITY_METHOD)

    k = corr_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(k))

    rows, cols = np.triu_indices(k, k=1)
    sim_weights = sim_matrix[rows, cols]
    corr_vals = corr_matrix[rows, cols]

    # Retém arestas com similaridade estritamente positiva (S > 1e-6, descartando apenas r == -1 exato)
    active_mask = sim_weights > 1e-6
    active_rows = rows[active_mask]
    active_cols = cols[active_mask]
    active_sim = sim_weights[active_mask]
    active_corr = corr_vals[active_mask]

    for u, v, s, r in zip(active_rows, active_cols, active_sim, active_corr):
        G.add_edge(int(u), int(v), weight=float(s), correlation=float(r), similarity=float(s))

    num_possible = (k * (k - 1)) // 2
    retained_edges = G.number_of_edges()
    degrees = [int(d) for _, d in G.degree()]
    isolated_nodes = sum(1 for d in degrees if d == 0)

    info = {
        "n_nodes": k,
        "total_possible_pairs": num_possible,
        "n_edges": retained_edges,
        "pct_edges_retained": (retained_edges / num_possible) * 100.0 if num_possible > 0 else 0.0,
        "isolated_nodes_count": isolated_nodes,
        "is_connected": nx.is_connected(G) if retained_edges > 0 and isolated_nodes == 0 else False,
        "mean_degree": float(np.mean(degrees)),
        "mean_correlation": float(np.mean(active_corr)) if len(active_corr) > 0 else 0.0,
        "mean_similarity": float(np.mean(active_sim)) if len(active_sim) > 0 else 0.0,
    }

    return G, info


def build_correlation_mst(
    corr_matrix: np.ndarray,
    sim_matrix: np.ndarray | None = None,
) -> tuple[nx.Graph, pd.DataFrame, dict]:
    """
    Cenário 2: Árvore Geradora Mínima (MST de Kruskal com Métrica Canônica de Mantegna, 1999).
    d_ij = sqrt(2 * (1 - r_ij)).
    Seleciona as N - 1 arestas de menor distância que conectam todos os nós sem formar ciclos.
    Atribui às arestas da árvore o peso de similaridade Louvain S_ij in [0, 1].
    """
    if sim_matrix is None:
        sim_matrix = correlation_to_similarity(corr_matrix, method=CORRELATION_SIMILARITY_METHOD)

    k = corr_matrix.shape[0]
    G_dist = nx.Graph()
    G_dist.add_nodes_from(range(k))

    rows, cols = np.triu_indices(k, k=1)
    r_vals = corr_matrix[rows, cols]
    d_vals = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - r_vals)))
    s_vals = sim_matrix[rows, cols]

    for u, v, d, r, s in zip(rows, cols, d_vals, r_vals, s_vals):
        G_dist.add_edge(int(u), int(v), distance=float(d), correlation=float(r), similarity=float(s))

    mst_dist = nx.minimum_spanning_tree(G_dist, weight="distance", algorithm="kruskal")

    G_mst = nx.Graph()
    G_mst.add_nodes_from(range(k))
    edge_records = []
    step = 1

    sorted_edges = sorted(
        mst_dist.edges(data=True),
        key=lambda x: x[2]["correlation"],
        reverse=True,
    )

    for u, v, data in sorted_edges:
        r = data["correlation"]
        d = data["distance"]
        s = data["similarity"]
        w = max(1e-6, float(s))
        G_mst.add_edge(u, v, weight=w, correlation=float(r), distance=float(d), similarity=float(s))
        edge_records.append({
            "kruskal_step": step,
            "node_u": int(u),
            "node_v": int(v),
            "correlation": float(r),
            "mantegna_distance": float(d),
            "similarity": float(s),
            "louvain_weight": float(w),
        })
        step += 1

    df_edges = pd.DataFrame(edge_records)

    num_possible = (k * (k - 1)) // 2
    retained = G_mst.number_of_edges()
    degrees = [d for _, d in G_mst.degree()]
    corr_weights = [d["correlation"] for _, _, d in G_mst.edges(data=True)]
    sim_weights = [d["similarity"] for _, _, d in G_mst.edges(data=True)]

    info = {
        "n_nodes": k,
        "n_edges": retained,
        "pct_edges_retained": (retained / num_possible) * 100.0 if num_possible > 0 else 0.0,
        "isolated_nodes_count": 0,
        "is_connected": nx.is_connected(G_mst),
        "is_tree": nx.is_tree(G_mst),
        "mean_degree": float(np.mean(degrees)),
        "max_degree": int(np.max(degrees)),
        "mean_correlation": float(np.mean(corr_weights)),
        "median_correlation": float(np.median(corr_weights)),
        "min_correlation": float(np.min(corr_weights)),
        "max_correlation": float(np.max(corr_weights)),
        "mean_similarity": float(np.mean(sim_weights)),
    }

    return G_mst, df_edges, info


def build_correlation_percentile_graph(
    corr_matrix: np.ndarray,
    percentile_cutoff: float = 25.0,
    sim_matrix: np.ndarray | None = None,
) -> tuple[nx.Graph, dict]:
    """
    Cenário 3A: Rede Filtrada por Percentil 25 Direto (Top 25% maiores similaridades / correlações).
    Retém exclusivamente as arestas com correlação/similaridade acima do limiar do percentil (100 - cutoff).
    Pesos das arestas utilizam a métrica de similaridade Louvain S_ij in [0, 1].
    """
    if sim_matrix is None:
        sim_matrix = correlation_to_similarity(corr_matrix, method=CORRELATION_SIMILARITY_METHOD)

    k = corr_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(k))

    rows, cols = np.triu_indices(k, k=1)
    r_vals = corr_matrix[rows, cols]
    s_vals = sim_matrix[rows, cols]

    threshold_r = float(np.percentile(r_vals, 100.0 - percentile_cutoff))
    threshold_s = float(np.percentile(s_vals, 100.0 - percentile_cutoff))
    mask = r_vals >= threshold_r

    sel_rows = rows[mask]
    sel_cols = cols[mask]
    sel_r = r_vals[mask]
    sel_s = s_vals[mask]

    for u, v, r, s in zip(sel_rows, sel_cols, sel_r, sel_s):
        w = max(1e-6, float(s))
        G.add_edge(int(u), int(v), weight=w, correlation=float(r), similarity=float(s))

    num_possible = (k * (k - 1)) // 2
    retained = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    isolated = sum(1 for d in degrees if d == 0)

    info = {
        "n_nodes": k,
        "percentile_cutoff": percentile_cutoff,
        "correlation_threshold": threshold_r,
        "similarity_threshold": threshold_s,
        "n_edges": retained,
        "pct_edges_retained": (retained / num_possible) * 100.0 if num_possible > 0 else 0.0,
        "isolated_nodes_count": isolated,
        "is_connected": nx.is_connected(G) if isolated == 0 and retained > 0 else False,
        "mean_degree": float(np.mean(degrees)),
        "mean_correlation": float(np.mean(sel_r)) if len(sel_r) > 0 else 0.0,
        "mean_similarity": float(np.mean(sel_s)) if len(sel_s) > 0 else 0.0,
    }

    return G, info


def build_correlation_optimized_graph(G_mst: nx.Graph, G_p25: nx.Graph) -> tuple[nx.Graph, dict]:
    """
    Cenário 3B: Rede Otimizada (União MST ∪ P25).
    E_otimizada = E_mst ∪ E_p25.
    Garante conectividade estrita (0 nós órfãos) e preserva as 25% arestas de maior similaridade.
    """
    k = G_mst.number_of_nodes()
    G_opt = nx.Graph()
    G_opt.add_nodes_from(range(k))

    for u, v, data in G_mst.edges(data=True):
        G_opt.add_edge(
            u, v,
            weight=data["weight"],
            correlation=data.get("correlation", 0.0),
            similarity=data.get("similarity", data["weight"]),
            in_mst=True,
        )

    for u, v, data in G_p25.edges(data=True):
        if not G_opt.has_edge(u, v):
            G_opt.add_edge(
                u, v,
                weight=data["weight"],
                correlation=data.get("correlation", 0.0),
                similarity=data.get("similarity", data["weight"]),
                in_mst=False,
            )

    num_possible = (k * (k - 1)) // 2
    retained = G_opt.number_of_edges()
    degrees = [d for _, d in G_opt.degree()]
    isolated = sum(1 for d in degrees if d == 0)
    corr_vals = [d["correlation"] for _, _, d in G_opt.edges(data=True)]
    sim_vals = [d["similarity"] for _, _, d in G_opt.edges(data=True)]

    info = {
        "n_nodes": k,
        "n_edges": retained,
        "pct_edges_retained": (retained / num_possible) * 100.0 if num_possible > 0 else 0.0,
        "isolated_nodes_count": isolated,
        "is_connected": nx.is_connected(G_opt),
        "mean_degree": float(np.mean(degrees)),
        "mean_correlation": float(np.mean(corr_vals)) if corr_vals else 0.0,
        "mean_similarity": float(np.mean(sim_vals)) if sim_vals else 0.0,
    }

    return G_opt, info


# ==============================================================================
# ALGORITMO DE LOUVAIN E REORDENAÇÃO
# ==============================================================================

def detect_louvain_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = LOUVAIN_RANDOM_SEED,
) -> tuple[list[list[int]], float]:
    """
    Executa o Algoritmo de Louvain Puro sobre o grafo ponderado de correlação.
    Retorna as comunidades (ordenadas por tamanho decrescente) e a modularidade Q.
    """
    if G.number_of_edges() == 0:
        return [[i] for i in range(G.number_of_nodes())], 0.0

    raw_communities = nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    modularity_q = nx.community.modularity(
        G,
        raw_communities,
        weight="weight",
        resolution=resolution,
    )

    communities = [sorted(list(c)) for c in raw_communities]
    communities.sort(key=len, reverse=True)
    return communities, float(modularity_q)


def reorder_matrix_by_communities(
    matrix: np.ndarray,
    communities: list[list[int]],
) -> tuple[np.ndarray, list[int], list[int], dict[int, int]]:
    """
    Reordena as linhas e colunas da matriz para agrupar as comunidades de Louvain.
    """
    ordered_indices: list[int] = []
    boundary_indices: list[int] = []
    node_to_comm: dict[int, int] = {}

    current_idx = 0
    for comm_id, comm in enumerate(communities):
        ordered_indices.extend(comm)
        current_idx += len(comm)
        boundary_indices.append(current_idx)
        for node in comm:
            node_to_comm[node] = comm_id

    reordered = matrix[np.ix_(ordered_indices, ordered_indices)]
    return reordered, ordered_indices, boundary_indices, node_to_comm


def compute_intra_inter_correlation(
    corr_matrix: np.ndarray,
    communities: list[list[int]],
    sim_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calcula a correlação média intra-comunidade e inter-comunidades,
    além da similaridade média correspondente.
    """
    node_to_comm = {}
    for c_id, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = c_id

    k = corr_matrix.shape[0]
    intra_vals = []
    inter_vals = []
    intra_sim = []
    inter_sim = []

    for i in range(k):
        for j in range(i + 1, k):
            r = corr_matrix[i, j]
            s = sim_matrix[i, j] if sim_matrix is not None else (1.0 + r) / 2.0
            if node_to_comm.get(i) == node_to_comm.get(j):
                intra_vals.append(r)
                intra_sim.append(s)
            else:
                inter_vals.append(r)
                inter_sim.append(s)

    mean_intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    mean_inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    mean_intra_s = float(np.mean(intra_sim)) if intra_sim else 0.0
    mean_inter_s = float(np.mean(inter_sim)) if inter_sim else 0.0

    return {
        "mean_intra_correlation": mean_intra,
        "mean_inter_correlation": mean_inter,
        "diff_intra_inter": mean_intra - mean_inter,
        "ratio_intra_inter": (mean_intra / mean_inter) if abs(mean_inter) > 1e-6 else 0.0,
        "mean_intra_similarity": mean_intra_s,
        "mean_inter_similarity": mean_inter_s,
    }


# ==============================================================================
# VISUALIZAÇÕES EM ALTA RESOLUÇÃO (300 DPI)
# ==============================================================================

def plot_raw_correlation_heatmap(
    corr_matrix: np.ndarray,
    output_png: Path,
    n_windows: int,
    feature: str = "speed_kmh",
) -> None:
    """Plota o Heatmap da matriz de correlação bruta na ordem original de índices."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)

    plt.figure(figsize=(10, 8), dpi=300)
    im = plt.imshow(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(f"Correlação de Pearson (r) — {fcfg['name_pt']}", fontsize=11, labelpad=10)

    plt.title(
        f"Matriz de Correlação de Pearson Bruta ({n_windows}x{n_windows})\n"
        f"Feature: {fcfg['name_pt']} (Ordem Original dos Trajetos 0 a {n_windows - 1})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(f"Índice da Janela de Treino j (0 a {n_windows - 1})", fontsize=11)
    plt.ylabel(f"Índice da Janela de Treino i (0 a {n_windows - 1})", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap de Correlação Bruta salvo em: {output_png.name}")


def plot_correlation_heatmap(
    corr_reordered: np.ndarray,
    boundary_indices: list[int],
    communities: list[list[int]],
    output_png: Path,
    n_windows: int,
    modularity_q: float,
    feature: str = "speed_kmh",
    stage_title: str = "Grafo Completo Positivo",
) -> None:
    """Plota o Heatmap da matriz de correlação reordenada pelas comunidades de Louvain."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)
    n_active_comm = sum(1 for c in communities if len(c) > 1)

    plt.figure(figsize=(11, 9), dpi=300)
    ax = plt.gca()

    im = ax.imshow(
        corr_reordered,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(f"Correlação de Pearson (r) — {fcfg['name_pt']}", fontsize=11, labelpad=10)

    for idx, b in enumerate(boundary_indices[:-1]):
        if len(communities[idx]) >= 2:
            ax.axvline(b - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.axhline(b - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    midpoints = []
    labels = []
    prev_b = 0
    for comm_id, b in enumerate(boundary_indices):
        size = b - prev_b
        mid = (prev_b + b - 1) / 2.0
        if size >= 2:
            midpoints.append(mid)
            labels.append(f"C{comm_id}\n(N={size})")
        prev_b = b

    ax.set_xticks(midpoints)
    ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax.set_yticks(midpoints)
    ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold")

    plt.title(
        f"Matriz de Correlação de Pearson Reordenada — {stage_title} ({n_windows}x{n_windows})\n"
        f"Feature: {fcfg['name_pt']} | {n_active_comm} Comunidades Principais | "
        f"Modularidade Q = {modularity_q:.4f}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("Comunidades Louvain (Correlação)", fontsize=10.5, labelpad=8)
    plt.ylabel("Comunidades Louvain (Correlação)", fontsize=10.5, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap salvo em: {output_png.name}")


def plot_correlation_synthesis(
    df_stages: pd.DataFrame,
    output_png: Path,
    feature: str = "speed_kmh",
) -> None:
    """
    Gera a figura de síntese comparativa dos 4 cenários na correlação de Pearson:
    Painel 1: Número de arestas retidas e % de preservação.
    Painel 2: Modularidade de Newman (Q).
    Painel 3: Número de comunidades detectadas pelo Louvain.
    Painel 4: Correlações médias intra-comunidade vs. inter-comunidade.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    stages = df_stages["cenario"].tolist()
    stage_short = ["1. Completo", "2. MST", "3A. P25", "3B. MST ∪ P25"]
    x = np.arange(len(stages))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # 1. Arestas retidas
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, df_stages["arestas"], color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Volume de Arestas Retidas na Rede", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_short, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Quantidade de Arestas", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, pct in zip(bars1, df_stages["pct_arestas"]):
        h = bar.get_height()
        ax1.annotate(
            f"{int(h):,d}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # 2. Modularidade Q
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df_stages["modularidade_q"], color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Modularidade de Newman (Q) — Louvain Puro", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_short, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Modularidade Q", fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"Q = {h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 3. Número de Comunidades
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, df_stages["comunidades_total"], color=colors, alpha=0.85, edgecolor="black")
    ax3.set_title("Número de Comunidades Detectadas", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(stage_short, fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Total de Comunidades", fontsize=10)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, n_act in zip(bars3, df_stages["comunidades_ativas"]):
        h = bar.get_height()
        ax3.annotate(
            f"{int(h)} totais\n({int(n_act)} ativas)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # 4. Correlação Intra vs Inter
    ax4 = axes[1, 1]
    w = 0.35
    ax4.bar(x - w / 2, df_stages["corr_intra_media"], width=w, label="Intra-Comunidade", color="#2ca02c", alpha=0.85, edgecolor="black")
    ax4.bar(x + w / 2, df_stages["corr_inter_media"], width=w, label="Inter-Comunidades", color="#d62728", alpha=0.85, edgecolor="black")
    ax4.set_title("Correlações Médias: Intra vs. Inter-Comunidades", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(stage_short, fontsize=9.5, fontweight="bold")
    ax4.set_ylabel("Correlação Média (r)", fontsize=10)
    ax4.legend(loc="upper right", fontsize=9.5)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle(
        f"Síntese Metodológica dos 4 Cenários Topológicos na Correlação de Pearson\n"
        f"Feature: {fcfg['name_pt']} (Treino 750x750)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Figura de Síntese Comparativa salva em: {output_png.name}")


# ==============================================================================
# CRUZAMENTO COMPARATIVO: CORRELAÇÃO VS DTW E K-MEANS
# ==============================================================================

def compare_scenarios_with_dtw_and_kmeans(
    df_all_clusters: pd.DataFrame,
    feature: str = "speed_kmh",
    is_pilot: bool = False,
) -> dict:
    """
    Cruza as comunidades de cada um dos 4 cenários de Correlação com os rótulos do
    K-Means e do DTW (mesma feature), computando ARI e NMI.
    """
    fcfg = get_feature_paths(feature)
    results = {}

    km_labeled_path = PROCESSED_FEATURES_DIR / "driving_windows_train_labeled.parquet"
    df_km = pd.read_parquet(km_labeled_path) if km_labeled_path.exists() and not is_pilot else None

    dtw_clusters_path = fcfg["louvain_pilot_parquet"] if is_pilot else fcfg["louvain_full_parquet"]
    df_dtw = pd.read_parquet(dtw_clusters_path) if dtw_clusters_path.exists() else None

    stage_cols = [
        ("Cenário 1: Completo", "corr_louvain_community"),
        ("Cenário 2: MST", "corr_mst_community"),
        ("Cenário 3A: P25", "corr_p25_community"),
        ("Cenário 3B: MST ∪ P25", "corr_p25_mst_community"),
    ]

    for stage_name, col in stage_cols:
        if col not in df_all_clusters.columns:
            continue
        stage_res = {}
        pred = df_all_clusters[col]

        # Vs K-Means
        if df_km is not None and "cluster_label" in df_km.columns:
            true_km = df_km["cluster_label"]
            ari_km = float(adjusted_rand_score(true_km, pred))
            nmi_km = float(normalized_mutual_info_score(true_km, pred))
            stage_res["kmeans"] = {"ari": ari_km, "nmi": nmi_km}

        # Vs DTW Louvain Completo
        if df_dtw is not None and "louvain_community" in df_dtw.columns:
            true_dtw = df_dtw["louvain_community"]
            ari_dtw = float(adjusted_rand_score(true_dtw, pred))
            nmi_dtw = float(normalized_mutual_info_score(true_dtw, pred))
            stage_res["dtw"] = {"ari": ari_dtw, "nmi": nmi_dtw}

        results[stage_name] = stage_res

    return results


# ==============================================================================
# ORQUESTRADOR DO PIPELINE COMPLETO
# ==============================================================================

def run_correlation_pipeline(
    feature: str = "speed_kmh",
    is_pilot: bool = False,
    n_points: int = 100,
    resolution: float = 1.0,
) -> dict:
    """
    Executa a cadeia metodológica unificada dos 4 cenários sobre a matriz de Correlação de Pearson:
    1. Grafo Completo + Louvain;
    2. MST (Kruskal com Mantegna) + Louvain;
    3. P25 Direto + Louvain;
    4. Rede Otimizada (MST ∪ P25) + Louvain;
    5. Síntese Comparativa dos 4 Cenários e Cruzamento com DTW e K-Means.
    """
    fcfg = get_feature_paths(feature)
    tag = "Piloto (100x100)" if is_pilot else "Completo (750x750)"
    n_windows = DTW_PILOT_SIZE if is_pilot else None

    print(f"\n{'='*95}")
    print(f"PIPELINE INTEGRADO DE CORRELAÇÃO DE PEARSON E REDES COMPLEXAS — {fcfg['name_pt'].upper()} [{tag}]")
    print(f"{'='*95}")

    # 1. Carrega e reamostra séries temporais
    t0 = time.time()
    X, file_stems = load_and_resample_series(feature=feature, n_windows=n_windows, n_points=n_points)
    k = X.shape[0]
    print(f"-> [PASSO 1/5] {k:,d} séries carregadas e reamostradas para grid de {n_points} pontos.")

    # 2. Calcula e salva matriz de correlação de Pearson e similaridade [0, 1]
    corr_matrix = compute_pearson_correlation_matrix(X)
    t_corr = time.time() - t0
    print(f"-> Matriz de Correlação de Pearson ({k}x{k}) calculada em {t_corr:.3f} s.")

    sim_matrix = correlation_to_similarity(corr_matrix, method=CORRELATION_SIMILARITY_METHOD)
    print(f"-> Métrica de Similaridade [0, 1] calculada via método '{CORRELATION_SIMILARITY_METHOD}'.")

    npy_path = fcfg["corr_pilot_npy"] if is_pilot else fcfg["corr_full_npy"]
    parquet_path = fcfg["corr_pilot_parquet"] if is_pilot else fcfg["corr_full_parquet"]
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, corr_matrix)
    pd.DataFrame(corr_matrix, index=file_stems, columns=file_stems).to_parquet(parquet_path)
    print(f"   Matriz salva em: {npy_path.name} e {parquet_path.name}")

    raw_heatmap_path = fcfg["fig_corr_pilot_heatmap"] if is_pilot else fcfg["fig_corr_full_heatmap"]
    plot_raw_correlation_heatmap(corr_matrix=corr_matrix, output_png=raw_heatmap_path, n_windows=k, feature=feature)

    stages_records = []

    # --------------------------------------------------------------------------
    # CENÁRIO 1: Grafo Completo de Similaridade [0, 1] + Louvain
    # --------------------------------------------------------------------------
    print(f"\n>>> [PASSO 2/5] Cenário 1: Grafo Completo de Similaridade [0, 1] (sem descarte de negativos) + Louvain...")
    G_full, info_full = build_full_correlation_graph(corr_matrix, sim_matrix=sim_matrix)
    comm_full, q_full = detect_louvain_communities(G_full, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    intra_inter_full = compute_intra_inter_correlation(corr_matrix, comm_full, sim_matrix=sim_matrix)
    n_act_full = sum(1 for c in comm_full if len(c) > 1)

    stages_records.append({
        "cenario": "1. Grafo Completo (Similaridade S ∈ [0, 1])",
        "arestas": G_full.number_of_edges(),
        "pct_arestas": info_full["pct_edges_retained"],
        "nos_orfaos": info_full["isolated_nodes_count"],
        "is_conexo": info_full["is_connected"],
        "modularidade_q": q_full,
        "comunidades_total": len(comm_full),
        "comunidades_ativas": n_act_full,
        "corr_intra_media": intra_inter_full["mean_intra_correlation"],
        "corr_inter_media": intra_inter_full["mean_inter_correlation"],
        "diff_intra_inter": intra_inter_full["diff_intra_inter"],
        "razao_intra_inter": intra_inter_full["ratio_intra_inter"],
    })

    corr_reord_full, _, b_full, node_to_full = reorder_matrix_by_communities(corr_matrix, comm_full)
    fig_full = fcfg["fig_corr_louvain_pilot_heatmap"] if is_pilot else fcfg["fig_corr_louvain_full_heatmap"]
    plot_correlation_heatmap(
        corr_reordered=corr_reord_full,
        boundary_indices=b_full,
        communities=comm_full,
        output_png=fig_full,
        n_windows=k,
        modularity_q=q_full,
        feature=feature,
        stage_title="Cenário 1 — Grafo Completo (Similaridade S ∈ [0, 1])",
    )

    # --------------------------------------------------------------------------
    # CENÁRIO 2: Árvore Geradora Mínima (MST de Kruskal com Mantegna) + Louvain
    # --------------------------------------------------------------------------
    print(f"\n>>> [PASSO 3/5] Cenário 2: Árvore Geradora Mínima (MST Kruskal - Métrica de Mantegna) + Louvain...")
    G_mst, df_mst_edges, info_mst = build_correlation_mst(corr_matrix, sim_matrix=sim_matrix)
    comm_mst, q_mst = detect_louvain_communities(G_mst, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    intra_inter_mst = compute_intra_inter_correlation(corr_matrix, comm_mst, sim_matrix=sim_matrix)
    n_act_mst = sum(1 for c in comm_mst if len(c) > 1)

    stages_records.append({
        "cenario": "2. Árvore Geradora Mínima (MST)",
        "arestas": G_mst.number_of_edges(),
        "pct_arestas": info_mst["pct_edges_retained"],
        "nos_orfaos": info_mst["isolated_nodes_count"],
        "is_conexo": info_mst["is_connected"],
        "modularidade_q": q_mst,
        "comunidades_total": len(comm_mst),
        "comunidades_ativas": n_act_mst,
        "corr_intra_media": intra_inter_mst["mean_intra_correlation"],
        "corr_inter_media": intra_inter_mst["mean_inter_correlation"],
        "diff_intra_inter": intra_inter_mst["diff_intra_inter"],
        "razao_intra_inter": intra_inter_mst["ratio_intra_inter"],
    })

    # Salva arestas da MST
    df_mst_edges.to_csv(fcfg["corr_mst_edges_csv"], index=False)
    df_mst_edges.to_parquet(fcfg["corr_mst_edges_parquet"], index=False)

    corr_reord_mst, _, b_mst, node_to_mst = reorder_matrix_by_communities(corr_matrix, comm_mst)
    plot_correlation_heatmap(
        corr_reordered=corr_reord_mst,
        boundary_indices=b_mst,
        communities=comm_mst,
        output_png=fcfg["fig_corr_mst_heatmap"],
        n_windows=k,
        modularity_q=q_mst,
        feature=feature,
        stage_title="Cenário 2 — Árvore Geradora Mínima (MST)",
    )

    # --------------------------------------------------------------------------
    # CENÁRIO 3A: Percentil 25 Direto de Correlação + Louvain
    # --------------------------------------------------------------------------
    print(f"\n>>> [PASSO 4/5] Cenário 3A: Filtragem por Percentil 25 Direto + Louvain...")
    G_p25, info_p25 = build_correlation_percentile_graph(corr_matrix, percentile_cutoff=25.0, sim_matrix=sim_matrix)
    comm_p25, q_p25 = detect_louvain_communities(G_p25, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    intra_inter_p25 = compute_intra_inter_correlation(corr_matrix, comm_p25, sim_matrix=sim_matrix)
    n_act_p25 = sum(1 for c in comm_p25 if len(c) > 1)

    stages_records.append({
        "cenario": "3A. Percentil 25 Direto (Sem MST)",
        "arestas": G_p25.number_of_edges(),
        "pct_arestas": info_p25["pct_edges_retained"],
        "nos_orfaos": info_p25["isolated_nodes_count"],
        "is_conexo": info_p25["is_connected"],
        "modularidade_q": q_p25,
        "comunidades_total": len(comm_p25),
        "comunidades_ativas": n_act_p25,
        "corr_intra_media": intra_inter_p25["mean_intra_correlation"],
        "corr_inter_media": intra_inter_p25["mean_inter_correlation"],
        "diff_intra_inter": intra_inter_p25["diff_intra_inter"],
        "razao_intra_inter": intra_inter_p25["ratio_intra_inter"],
    })

    corr_reord_p25, _, b_p25, node_to_p25 = reorder_matrix_by_communities(corr_matrix, comm_p25)
    plot_correlation_heatmap(
        corr_reordered=corr_reord_p25,
        boundary_indices=b_p25,
        communities=comm_p25,
        output_png=fcfg["fig_corr_p25_direct_heatmap"],
        n_windows=k,
        modularity_q=q_p25,
        feature=feature,
        stage_title="Cenário 3A — Percentil 25 Direto",
    )

    # --------------------------------------------------------------------------
    # CENÁRIO 3B: Rede Otimizada (MST ∪ P25) + Louvain
    # --------------------------------------------------------------------------
    print(f"\n>>> [PASSO 5/5] Cenário 3B: Rede Otimizada (União MST ∪ P25) + Louvain...")
    G_opt, info_opt = build_correlation_optimized_graph(G_mst, G_p25)
    comm_opt, q_opt = detect_louvain_communities(G_opt, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    intra_inter_opt = compute_intra_inter_correlation(corr_matrix, comm_opt, sim_matrix=sim_matrix)
    n_act_opt = sum(1 for c in comm_opt if len(c) > 1)

    stages_records.append({
        "cenario": "3B. Rede Otimizada (MST ∪ P25)",
        "arestas": G_opt.number_of_edges(),
        "pct_arestas": info_opt["pct_edges_retained"],
        "nos_orfaos": info_opt["isolated_nodes_count"],
        "is_conexo": info_opt["is_connected"],
        "modularidade_q": q_opt,
        "comunidades_total": len(comm_opt),
        "comunidades_ativas": n_act_opt,
        "corr_intra_media": intra_inter_opt["mean_intra_correlation"],
        "corr_inter_media": intra_inter_opt["mean_inter_correlation"],
        "diff_intra_inter": intra_inter_opt["diff_intra_inter"],
        "razao_intra_inter": intra_inter_opt["ratio_intra_inter"],
    })

    corr_reord_opt, _, b_opt, node_to_opt = reorder_matrix_by_communities(corr_matrix, comm_opt)
    plot_correlation_heatmap(
        corr_reordered=corr_reord_opt,
        boundary_indices=b_opt,
        communities=comm_opt,
        output_png=fcfg["fig_corr_p25_mst_heatmap"],
        n_windows=k,
        modularity_q=q_opt,
        feature=feature,
        stage_title="Cenário 3B — Rede Otimizada (MST ∪ P25)",
    )

    # --------------------------------------------------------------------------
    # SALVAMENTO DOS CLUSTERS ATRIBUÍDOS POR CENÁRIO
    # --------------------------------------------------------------------------
    train_feat_path = PROCESSED_FEATURES_DIR / "driving_windows_train_features.parquet"
    if train_feat_path.exists() and not is_pilot:
        df_clusters = pd.read_parquet(train_feat_path).iloc[:k].copy()
    else:
        df_clusters = pd.DataFrame({"window_stem": file_stems})

    df_clusters["corr_louvain_community"] = [node_to_full[i] for i in range(k)]
    df_clusters["corr_mst_community"] = [node_to_mst[i] for i in range(k)]
    df_clusters["corr_p25_community"] = [node_to_p25[i] for i in range(k)]
    df_clusters["corr_p25_mst_community"] = [node_to_opt[i] for i in range(k)]

    # Salva arquivos de clusters para cada cenário
    # Cenário 1
    df_clusters.to_csv(fcfg["corr_louvain_pilot_csv"] if is_pilot else fcfg["corr_louvain_full_csv"], index=False)
    df_clusters.to_parquet(fcfg["corr_louvain_pilot_parquet"] if is_pilot else fcfg["corr_louvain_full_parquet"], index=False)
    # Cenário 2
    df_clusters.to_csv(fcfg["corr_mst_louvain_csv"], index=False)
    df_clusters.to_parquet(fcfg["corr_mst_louvain_parquet"], index=False)
    # Cenário 3A
    df_clusters.to_csv(fcfg["corr_p25_direct_csv"], index=False)
    df_clusters.to_parquet(fcfg["corr_p25_direct_parquet"], index=False)
    # Cenário 3B
    df_clusters.to_csv(fcfg["corr_p25_mst_csv"], index=False)
    df_clusters.to_parquet(fcfg["corr_p25_mst_parquet"], index=False)

    # --------------------------------------------------------------------------
    # CONSOLIDAÇÃO DA SÍNTESE COMPARATIVA DOS 4 CENÁRIOS
    # --------------------------------------------------------------------------
    df_stages = pd.DataFrame(stages_records)
    print("\n" + "=" * 105)
    print(f"TABELA COMPARATIVA DOS 4 CENÁRIOS TOPOLÓGICOS (CORRELAÇÃO DE PEARSON — {fcfg['name_pt'].upper()}):")
    print("=" * 105)
    print(df_stages.to_string(index=False))
    print("=" * 105)

    df_stages.to_csv(fcfg["corr_all_stages_comparison_csv"], index=False)
    with open(fcfg["corr_all_stages_comparison_json"], "w", encoding="utf-8") as f:
        json.dump(df_stages.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print(f"-> Tabela comparativa salva em: {fcfg['corr_all_stages_comparison_csv'].name}")

    # Plota a Figura de Síntese
    plot_correlation_synthesis(df_stages, fcfg["fig_corr_final_comparison_synthesis"], feature=feature)

    # --------------------------------------------------------------------------
    # CRUZAMENTO COM DTW E K-MEANS PARA CADA CENÁRIO
    # --------------------------------------------------------------------------
    comp_results = compare_scenarios_with_dtw_and_kmeans(df_clusters, feature=feature, is_pilot=is_pilot)
    print("\n" + "-" * 75)
    print("CRUZAMENTO DE CONCORDÂNCIA (ARI / NMI) CONTRA DTW E K-MEANS:")
    print("-" * 75)
    for stg, metrics in comp_results.items():
        km_str = f"K-Means (ARI={metrics.get('kmeans',{}).get('ari',0.0):.4f}, NMI={metrics.get('kmeans',{}).get('nmi',0.0):.4f})" if 'kmeans' in metrics else ""
        dtw_str = f"DTW (ARI={metrics.get('dtw',{}).get('ari',0.0):.4f}, NMI={metrics.get('dtw',{}).get('nmi',0.0):.4f})" if 'dtw' in metrics else ""
        print(f"• {stg:<28} | {km_str} | {dtw_str}")
    print("-" * 75)

    # Relatório final consolidado
    summary_report = {
        "feature": feature,
        "is_pilot": is_pilot,
        "n_windows": k,
        "stages_comparison": df_stages.to_dict(orient="records"),
        "cross_comparisons": comp_results,
    }

    json_path = fcfg["corr_vs_dtw_comparison_json"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"-> Relatório consolidado salvo em: {json_path.name}")

    return summary_report


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Unificado de Correlação de Pearson, MST e P25 com Louvain."
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=["speed_kmh", "accel_ms2", "power_pos_kw", "all"],
        help="Feature física a analisar (speed_kmh, accel_ms2, power_pos_kw ou all).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Executa o teste piloto com as primeiras 100 janelas.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=100,
        help="Número de pontos do grid uniforme de interpolação temporal (padrão: 100).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Parâmetro de resolução do Algoritmo de Louvain (padrão: 1.0).",
    )
    args = parser.parse_args()

    features_to_run = (
        ["speed_kmh", "accel_ms2", "power_pos_kw"]
        if args.feature == "all"
        else [args.feature]
    )

    for feat in features_to_run:
        run_correlation_pipeline(
            feature=feat,
            is_pilot=args.pilot,
            n_points=args.points,
            resolution=args.resolution,
        )


if __name__ == "__main__":
    main()
