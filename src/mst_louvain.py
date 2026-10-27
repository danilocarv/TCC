"""
Módulo da Etapa 2: Detecção de Comunidades via Algoritmo de Louvain 100% Puro sobre a MST.

Opera de forma estritamente sequencial após a Etapa 1 (src/mst_builder.py):
1. Carrega exclusivamente a Árvore Geradora Mínima (MST) gerada na Etapa 1;
2. Calcula o peso de afinidade contínua W_ij = exp(-D_ij / sigma) apenas para as N - 1 arestas da MST;
3. Executa o Algoritmo de Louvain 100% puro (resolution=1.0, sem induzir k, sem usar variáveis externas
   e sem aplicar rótulos heurísticos forçados) sobre a estrutura da MST;
4. Gera:
   - Heatmap da matriz DTW reordenada pelas comunidades descobertas na MST;
   - Visualização topológica da MST colorida pelas comunidades do Louvain e pela velocidade média;
   - Relatório e arquivos comparativos (.csv e .json) entre Grafo Completo, MST Pura e K-Means.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.config import (
    PROCESSED_TRAIN_FEATURES_PARQUET,
    PROCESSED_CLUSTERS_PARQUET,
    DTW_PILOT_SIZE,
    DTW_PILOT_MATRIX_NPY,
    DTW_SPEED_MATRIX_NPY,
    LOUVAIN_PILOT_CLUSTERS_PARQUET,
    LOUVAIN_FULL_CLUSTERS_PARQUET,
    LOUVAIN_RANDOM_SEED,
    MST_PILOT_EDGES_PARQUET,
    MST_FULL_EDGES_PARQUET,
    LOUVAIN_MST_PILOT_CLUSTERS_CSV,
    LOUVAIN_MST_PILOT_CLUSTERS_PARQUET,
    FIG_LOUVAIN_MST_PILOT_HEATMAP,
    FIG_LOUVAIN_MST_TOPOLOGY_PILOT,
    LOUVAIN_MST_FULL_CLUSTERS_CSV,
    LOUVAIN_MST_FULL_CLUSTERS_PARQUET,
    FIG_LOUVAIN_MST_FULL_HEATMAP,
    FIG_LOUVAIN_MST_TOPOLOGY_FULL,
    LOUVAIN_MST_COMPARISON_JSON,
    LOUVAIN_MST_COMPARISON_CSV,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)
from src.louvain_clustering import (
    detect_louvain_communities,
    reorder_dtw_matrix,
    compute_pure_dtw_community_metrics,
)


def load_mst_affinity_graph(
    mst_edges_parquet: Path,
    dist_matrix: np.ndarray,
    sigma: float | None = None,
) -> tuple[nx.Graph, float, pd.DataFrame]:
    """
    Carrega a Árvore Geradora Mínima (MST) salva na Etapa 1 e atribui às suas N - 1 arestas
    o peso de afinidade W_ij = exp(-D_ij / sigma) para execução do algoritmo de Louvain.
    """
    if not mst_edges_parquet.exists():
        raise FileNotFoundError(
            f"Arquivo da MST não encontrado em {mst_edges_parquet}. "
            "Execute primeiro a Etapa 1 ('python src/mst_builder.py')."
        )

    n = dist_matrix.shape[0]
    if sigma is None:
        off_diag = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        sigma = float(np.median(off_diag))
        if sigma <= 0.0:
            sigma = 1.0

    df_edges = pd.read_parquet(mst_edges_parquet)
    T_mst = nx.Graph()
    T_mst.add_nodes_from(range(n))

    for _, row in df_edges.iterrows():
        u = int(row["node_u"])
        v = int(row["node_v"])
        dtw_dist = float(row["dtw_distance"])
        affinity_w = float(np.exp(-dtw_dist / sigma))
        T_mst.add_edge(u, v, weight=affinity_w, dtw_distance=dtw_dist)

    assert T_mst.number_of_nodes() == n, "Número de nós na MST carregada diverge da matriz DTW!"
    assert T_mst.number_of_edges() == n - 1, "Número de arestas na MST deve ser exatamente N - 1!"
    assert nx.number_of_selfloops(T_mst) == 0, "A MST não pode conter auto-laços!"
    assert nx.is_tree(T_mst), "O grafo carregado deve ser estritamente uma árvore!"

    return T_mst, sigma, df_edges


def plot_mst_louvain_heatmap(
    matrix_reordered: np.ndarray,
    boundary_indices: list[int],
    communities: list[list[int]],
    output_png: Path,
    n_windows: int,
    sigma: float,
    modularity_q: float,
    feature: str = "speed_kmh",
):
    """
    Plota o mapa de calor (Heatmap) da matriz DTW reordenada estritamente pelas
    comunidades descobertas pelo Louvain Puro sobre a Árvore Geradora Mínima (MST).
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)
    n_comm = len(communities)

    plt.figure(figsize=(11, 9), dpi=300)
    ax = plt.gca()

    im = ax.imshow(
        matrix_reordered,
        cmap="viridis",
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(f"Distância DTW Acumulada sobre {fcfg['name_pt']} ({fcfg['unit']})", fontsize=11, labelpad=10)

    # Desenha as linhas divisórias entre comunidades
    line_alpha = 0.85 if n_comm <= 15 else 0.45
    line_width = 1.2 if n_comm <= 15 else 0.6
    for b in boundary_indices[:-1]:
        ax.axvline(b - 0.5, color="red", linestyle="--", linewidth=line_width, alpha=line_alpha)
        ax.axhline(b - 0.5, color="red", linestyle="--", linewidth=line_width, alpha=line_alpha)

    # Rotula todas as comunidades nos eixos X e Y (ajustando tamanho de fonte conforme a quantidade)
    midpoints = []
    labels = []
    prev_b = 0
    for comm_id, b in enumerate(boundary_indices):
        size = b - prev_b
        mid = (prev_b + b - 1) / 2.0
        midpoints.append(mid)
        if n_comm <= 15:
            labels.append(f"C{comm_id}\n(N={size})")
        else:
            labels.append(f"C{comm_id}")
        prev_b = b

    tick_fontsize = 9 if n_comm <= 15 else 6.5
    ax.set_xticks(midpoints)
    ax.set_xticklabels(labels, fontsize=tick_fontsize, fontweight="bold", rotation=90 if n_comm > 15 else 0)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(labels, fontsize=tick_fontsize, fontweight="bold")

    plt.title(
        f"Matriz DTW Reordenada pelas Comunidades de Louvain sobre a MST ({n_windows}x{n_windows})\n"
        f"Louvain 100% Puro na Árvore Geradora Mínima ({n_windows - 1} arestas) | "
        f"{n_comm} Comunidades Naturais | Modularidade Q = {modularity_q:.4f}",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("Comunidades de Louvain na MST (Ordem Original Interna Preservada)", fontsize=11, labelpad=8)
    plt.ylabel("Comunidades de Louvain na MST (Ordem Original Interna Preservada)", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap MST + Louvain salvo em: {output_png.name} ({output_png})")


def plot_mst_louvain_topology(
    T_mst: nx.Graph,
    node_to_comm: dict[int, int],
    n_communities: int,
    df_features: pd.DataFrame,
    modularity_q: float,
    output_png: Path,
    is_pilot: bool,
    feature: str = "speed_kmh",
):
    """
    Plota painel duplo (300 DPI) da topologia da Árvore Geradora Mínima (MST):
    - Painel A: Nós coloridos pelas Comunidades Naturais de Louvain na MST (destacando arestas
      intra-comunidade vs. arestas de corte entre comunidades);
    - Painel B: Nós coloridos pelo gradiente físico correspondente à feature analisada.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)
    n = T_mst.number_of_nodes()

    fig, (ax_comm, ax_speed) = plt.subplots(1, 2, figsize=(16.5, 7.8), dpi=300)

    if is_pilot:
        pos = nx.kamada_kawai_layout(T_mst, weight=None)
        node_size = 65
        intra_width = 1.4
        inter_width = 0.9
    else:
        pos = nx.spring_layout(T_mst, k=1.8 / np.sqrt(n), iterations=120, seed=42, weight=None)
        node_size = 18
        intra_width = 0.8
        inter_width = 0.5

    # Separa arestas intra-comunidade vs. arestas de fronteira (inter-comunidade)
    intra_edges = [(u, v) for u, v in T_mst.edges() if node_to_comm[u] == node_to_comm[v]]
    inter_edges = [(u, v) for u, v in T_mst.edges() if node_to_comm[u] != node_to_comm[v]]

    # Cores perceptualmente ricas e distintas para cada comunidade
    import matplotlib as mpl
    turbo_cmap = mpl.colormaps["turbo"]
    comm_colors = turbo_cmap(np.linspace(0.02, 0.98, n_communities))
    node_colors = [comm_colors[node_to_comm[i]] for i in range(n)]

    if feature == "accel_ms2":
        grad_values = df_features["mean_pos_accel_ms2"].to_numpy(dtype=np.float64)
        cbar_title = "Aceleração Média Positiva (m/s²)"
        panel_feat_name = "Aceleração"
    elif feature in ["power_kw", "power_pos_kw"]:
        grad_values = df_features["mean_power_kw"].to_numpy(dtype=np.float64)
        cbar_title = "Potência Média da Janela (kW)"
        panel_feat_name = "Potência Elétrica"
    else:
        grad_values = df_features["mean_speed_kmh"].to_numpy(dtype=np.float64)
        cbar_title = "Velocidade Média da Janela (km/h)"
        panel_feat_name = "Velocidade"

    # Painel A: Comunidades de Louvain na MST
    nx.draw_networkx_edges(
        T_mst,
        pos=pos,
        edgelist=intra_edges,
        ax=ax_comm,
        edge_color="#495057",
        width=intra_width,
        alpha=0.55 if is_pilot else 0.35,
        label=f"Arestas Intra-Comunidade ({len(intra_edges)})",
    )
    if inter_edges:
        nx.draw_networkx_edges(
            T_mst,
            pos=pos,
            edgelist=inter_edges,
            ax=ax_comm,
            edge_color="#e63946",
            style="dashed",
            width=inter_width,
            alpha=0.85,
            label=f"Fronteiras entre Comunidades ({len(inter_edges)} cortes)",
        )

    nx.draw_networkx_nodes(
        T_mst,
        pos=pos,
        ax=ax_comm,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.3,
        edgecolors="black",
    )

    # Adiciona rótulos identificadores (C0, C1, ...) no centro de massa (medoide) de cada comunidade
    for c_id in range(n_communities):
        nodes_in_c = [i for i in range(n) if node_to_comm[i] == c_id]
        if not nodes_in_c:
            continue
        xs = np.array([pos[i][0] for i in nodes_in_c])
        ys = np.array([pos[i][1] for i in nodes_in_c])
        # Encontra o nó da comunidade mais próximo do centro geométrico para ancorar o rótulo
        mean_x, mean_y = np.mean(xs), np.mean(ys)
        dists = (xs - mean_x) ** 2 + (ys - mean_y) ** 2
        best_node = nodes_in_c[int(np.argmin(dists))]
        lbl_x, lbl_y = pos[best_node][0], pos[best_node][1]

        badge_fontsize = 6.0 if n_communities > 15 else 8.5
        ax_comm.text(
            lbl_x,
            lbl_y,
            f"C{c_id}",
            fontsize=badge_fontsize,
            fontweight="bold",
            color="#111111",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=comm_colors[c_id],
                alpha=0.95,
                linewidth=1.2,
            ),
            zorder=15,
        )

    ax_comm.legend(loc="lower left", fontsize=8, frameon=True)

    ax_comm.set_title(
        f"Partição de Comunidades de Louvain sobre a MST\n"
        f"{n_communities} Comunidades Naturais Identificadas (C0 a C{n_communities - 1}) | "
        f"Modularidade Q = {modularity_q:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_comm.axis("off")

    # Painel B: Gradiente Físico Observado a Posteriori
    nx.draw_networkx_edges(
        T_mst,
        pos=pos,
        ax=ax_speed,
        edge_color="#6c757d",
        width=intra_width * 0.85,
        alpha=0.35 if is_pilot else 0.22,
    )
    sc_grad = nx.draw_networkx_nodes(
        T_mst,
        pos=pos,
        ax=ax_speed,
        node_size=node_size,
        node_color=grad_values,
        cmap="plasma",
        alpha=0.92,
        linewidths=0.3,
        edgecolors="black",
    )
    cbar = plt.colorbar(sc_grad, ax=ax_speed, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_title, fontsize=10)

    ax_speed.set_title(
        f"Distribuição Física de {panel_feat_name} ao Longo dos Ramos da MST\n"
        "(Gradiente Contínuo Observado a Posteriori sem Indução no Algoritmo)",
        fontsize=11,
        fontweight="bold",
    )
    ax_speed.axis("off")

    plt.suptitle(
        f"ETAPA 2: Topologia da Árvore Geradora Mínima (MST) e Comunidades Naturais de Louvain ({n} Nós)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Topologia MST + Louvain salva em: {output_png.name} ({output_png})")


def print_mst_louvain_report(
    df_dtw_metrics: pd.DataFrame,
    df_features: pd.DataFrame,
    modularity_q: float,
    sigma: float,
    resolution: float,
    n_edges: int,
):
    """
    Exibe o relatório completo da Etapa 2 (Louvain Puro sobre a MST):
    - Parte 1: Métricas Topológicas Puras da Matriz DTW e da MST;
    - Parte 2: Estatísticas descritivas observadas a posteriori nas comunidades.
    """
    n_comm = len(df_dtw_metrics)
    print("\n" + "=" * 96)
    print("   ETAPA 2 - PARTE 1: RESULTADO DO LOUVAIN 100% PURO SOBRE A ÁRVORE GERADORA MÍNIMA (MST)")
    print("=" * 96)
    print(f"-> Grafo de Entrada: MST Pura ({n_edges} arestas) | sigma = {sigma:.2f} km/h | Resolução = {resolution}")
    print(f"-> Modularidade de Newman na MST (Q) = {modularity_q:.4f}")
    print(f"-> Comunidades Naturais Detectadas (Sem Indução): {n_comm}")
    print("-" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'Proporção':<10} | "
        f"{'DTW Interno Médio':<19} | {'DTW Externo Médio':<19} | {'Razão Int/Ext':<13}"
    )
    print("-" * 96)

    # Exibe todas se <= 15 comunidades, ou top 15 maiores comunidades para leitura limpa no console
    df_sorted = df_dtw_metrics.sort_values(by="count", ascending=False)
    display_df = df_dtw_metrics if n_comm <= 15 else df_sorted.head(15)

    for _, row in display_df.iterrows():
        ratio = row["intra_dtw_mean"] / row["inter_dtw_mean"] if row["inter_dtw_mean"] > 0 else 0.0
        print(
            f"C{int(row['community_id']):<11} | "
            f"{int(row['count']):<8} | "
            f"{row['percentage']:>8.1f}%  | "
            f"{row['intra_dtw_mean']:>19.2f} | "
            f"{row['inter_dtw_mean']:>19.2f} | "
            f"{ratio:>13.3f}"
        )
    if n_comm > 15:
        print(f"   (... exibindo as 15 maiores comunidades de um total de {n_comm} comunidades naturais ...)")

    print("\n" + "=" * 96)
    print("   ETAPA 2 - PARTE 2: MÉDIAS FÍSICAS OBSERVADAS A POSTERIORI POR COMUNIDADE DA MST")
    print("=" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'v_méd (km/h)':<12} | {'std_v (km/h)':<12} | "
        f"{'a_méd (m/s²)':<12} | {'Fren/min':<9} | {'Acel/min':<9}"
    )
    print("-" * 96)

    comm_ids_to_show = display_df["community_id"].astype(int).tolist()
    for comm_id in comm_ids_to_show:
        sub = df_features[df_features["mst_louvain_community"] == comm_id]
        print(
            f"C{int(comm_id):<11} | "
            f"{len(sub):<8} | "
            f"{sub['mean_speed_kmh'].mean():>12.2f} | "
            f"{sub['std_speed_kmh'].mean():>12.2f} | "
            f"{sub['mean_pos_accel_ms2'].mean():>12.2f} | "
            f"{sub['hard_braking_rate_min'].mean():>9.2f} | "
            f"{sub['rapid_accel_rate_min'].mean():>9.2f}"
        )
    print("=" * 96 + "\n")


def save_comparison_metrics(
    is_pilot: bool,
    n_windows: int,
    n_mst_edges: int,
    sigma: float,
    mst_modularity_q: float,
    df_mst_metrics: pd.DataFrame,
    df_features: pd.DataFrame,
    dist_matrix: np.ndarray,
    feature: str = "speed_kmh",
):
    """
    Consolida e salva a comparação direta entre:
    1. Louvain no Grafo Completo (N*(N-1)/2 arestas);
    2. Louvain na Árvore Geradora Mínima Pura (N-1 arestas);
    3. Cruzamento estatístico com os clusters prévios (Louvain Completo e K-Means).
    """
    fcfg = get_feature_paths(feature)
    full_edges = (n_windows * (n_windows - 1)) // 2
    weighted_intra_mst = float(
        np.sum(df_mst_metrics["intra_dtw_mean"] * df_mst_metrics["count"]) / n_windows
    )
    weighted_inter_mst = float(
        np.sum(df_mst_metrics["inter_dtw_mean"] * df_mst_metrics["count"]) / n_windows
    )

    # Carrega o resultado do Louvain no Grafo Completo para comparação "Antes vs. Depois"
    full_louvain_pq = fcfg["louvain_pilot_parquet"] if is_pilot else fcfg["louvain_full_parquet"]
    full_info = {}
    if full_louvain_pq.exists():
        df_full_louvain = pd.read_parquet(full_louvain_pq)
        full_comms = sorted(df_full_louvain["louvain_community"].unique())
        full_comm_list = [
            df_full_louvain.index[df_full_louvain["louvain_community"] == c].tolist()
            for c in full_comms
        ]
        # Reconstrói modularidade do grafo completo
        sim_mat = np.exp(-dist_matrix / sigma)
        np.fill_diagonal(sim_mat, 0.0)
        G_full_sim = nx.from_numpy_array(sim_mat)
        q_full = float(nx.community.modularity(G_full_sim, full_comm_list, weight="weight"))
        df_full_dtw_m = compute_pure_dtw_community_metrics(dist_matrix, full_comm_list)
        w_intra_full = float(
            np.sum(df_full_dtw_m["intra_dtw_mean"] * df_full_dtw_m["count"]) / n_windows
        )
        w_inter_full = float(
            np.sum(df_full_dtw_m["inter_dtw_mean"] * df_full_dtw_m["count"]) / n_windows
        )
        full_info = {
            "n_edges": int(full_edges),
            "n_communities": int(len(full_comms)),
            "modularity_q": q_full,
            "weighted_intra_dtw_mean": w_intra_full,
            "weighted_inter_dtw_mean": w_inter_full,
            "intra_inter_ratio": float(w_intra_full / w_inter_full) if w_inter_full > 0 else 0.0,
        }

    mst_info = {
        "n_edges": int(n_mst_edges),
        "n_communities": int(len(df_mst_metrics)),
        "modularity_q": float(mst_modularity_q),
        "weighted_intra_dtw_mean": weighted_intra_mst,
        "weighted_inter_dtw_mean": weighted_inter_mst,
        "intra_inter_ratio": float(weighted_intra_mst / weighted_inter_mst) if weighted_inter_mst > 0 else 0.0,
    }

    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"
    comp_json = fcfg["louvain_mst_comparison_json"]
    comp_csv = fcfg["louvain_mst_comparison_csv"]
    comp_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if comp_json.exists():
        try:
            payload = json.loads(comp_json.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload[mode_key] = {
        "n_nodes": int(n_windows),
        "sigma_kmh": float(sigma),
        "full_graph_louvain": full_info,
        "mst_pure_louvain": mst_info,
        "modularity_gain_factor": (
            float(mst_info["modularity_q"] / full_info["modularity_q"])
            if full_info and full_info.get("modularity_q", 0) > 0
            else None
        ),
    }
    comp_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Salva também tabela CSV comparativa direta das comunidades da MST
    df_summary = df_mst_metrics.copy()
    kinematic_means = (
        df_features.groupby("mst_louvain_community")[
            [
                "mean_speed_kmh",
                "std_speed_kmh",
                "max_speed_kmh",
                "mean_pos_accel_ms2",
                "hard_braking_rate_min",
                "rapid_accel_rate_min",
                "mean_power_kw",
            ]
        ]
        .mean()
        .reset_index()
        .rename(columns={"mst_louvain_community": "community_id"})
    )
    df_summary = df_summary.merge(kinematic_means, on="community_id", how="left")
    if not is_pilot:
        df_summary.to_csv(comp_csv, index=False, encoding="utf-8")


def run_mst_louvain(
    is_pilot: bool = True,
    resolution: float = 1.0,
    feature: str = "speed_kmh",
) -> tuple[pd.DataFrame, float]:
    """
    Executa a Etapa 2: Algoritmo de Louvain 100% puro sobre a Árvore Geradora Mínima (MST).
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 88)
    print(
        f"INICIANDO ETAPA 2: LOUVAIN 100% PURO SOBRE A MST [{fcfg['name_pt'].upper()}] - MODO {mode_str}"
    )
    print("=" * 88)

    dtw_npy_path = fcfg["dtw_pilot_npy"] if is_pilot else fcfg["dtw_full_npy"]
    mst_edges_pq = fcfg["mst_pilot_edges_parquet"] if is_pilot else fcfg["mst_full_edges_parquet"]

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]

    print(f"-> Carregando Árvore Geradora Mínima da Etapa 1: {mst_edges_pq.name}...")
    start_t = time.time()
    T_mst, sigma, df_edges = load_mst_affinity_graph(mst_edges_pq, dist_matrix)
    print(
        f"   MST carregada: {T_mst.number_of_nodes()} nós e {T_mst.number_of_edges()} arestas "
        f"(sigma = {sigma:.2f} {fcfg['unit']})."
    )

    print(f"-> Executando Algoritmo de Louvain 100% puro na MST (resolução={resolution}, sem indução de k)...")
    communities, modularity_q = detect_louvain_communities(
        T_mst, resolution=resolution, seed=LOUVAIN_RANDOM_SEED
    )
    elapsed_s = time.time() - start_t
    print(
        f"   Louvain na MST concluído em {elapsed_s:.3f}s: "
        f"{len(communities)} comunidades naturais detectadas!"
    )
    print(f"   Modularidade de Newman na MST (Q) = {modularity_q:.4f}")

    # Reordena a matriz DTW apenas agrupando os nós da mesma comunidade da MST
    reordered_dist, ordered_indices, boundary_indices, node_to_comm = reorder_dtw_matrix(
        dist_matrix, communities
    )

    # Calcula métricas puras da matriz DTW para cada comunidade da MST
    df_dtw_metrics = compute_pure_dtw_community_metrics(dist_matrix, communities)

    # Anexa o ID da comunidade da MST ao dataset
    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    df_features["mst_louvain_community"] = [node_to_comm[i] for i in range(n_windows)]

    # Gera Heatmap reordenado pelas comunidades da MST
    heatmap_path = (
        fcfg["fig_louvain_mst_pilot_heatmap"] if is_pilot else fcfg["fig_louvain_mst_full_heatmap"]
    )
    plot_mst_louvain_heatmap(
        matrix_reordered=reordered_dist,
        boundary_indices=boundary_indices,
        communities=communities,
        output_png=heatmap_path,
        n_windows=n_windows,
        sigma=sigma,
        modularity_q=modularity_q,
        feature=feature,
    )

    # Gera visualização topológica da MST colorida pelas comunidades de Louvain
    topology_path = (
        fcfg["fig_louvain_mst_topology_pilot"] if is_pilot else fcfg["fig_louvain_mst_topology_full"]
    )
    plot_mst_louvain_topology(
        T_mst=T_mst,
        node_to_comm=node_to_comm,
        n_communities=len(communities),
        df_features=df_features,
        modularity_q=modularity_q,
        output_png=topology_path,
        is_pilot=is_pilot,
        feature=feature,
    )

    print_mst_louvain_report(
        df_dtw_metrics=df_dtw_metrics,
        df_features=df_features,
        modularity_q=modularity_q,
        sigma=sigma,
        resolution=resolution,
        n_edges=T_mst.number_of_edges(),
    )

    # Salva datasets de comunidades da MST e métricas comparativas
    out_csv = fcfg["louvain_mst_pilot_csv"] if is_pilot else fcfg["louvain_mst_full_csv"]
    out_pq = fcfg["louvain_mst_pilot_parquet"] if is_pilot else fcfg["louvain_mst_full_parquet"]
    df_features.to_csv(out_csv, index=False, encoding="utf-8")
    df_features.to_parquet(out_pq, index=False)

    save_comparison_metrics(
        is_pilot=is_pilot,
        n_windows=n_windows,
        n_mst_edges=T_mst.number_of_edges(),
        sigma=sigma,
        mst_modularity_q=modularity_q,
        df_mst_metrics=df_dtw_metrics,
        df_features=df_features,
        dist_matrix=dist_matrix,
        feature=feature,
    )

    print("-> Artefatos da Etapa 2 (Louvain Puro na MST) salvos com sucesso:")
    print(f"   Clusters CSV:     {out_csv.name}")
    print(f"   Clusters Parquet: {out_pq.name}")
    print(f"   Comparativo JSON: {fcfg['louvain_mst_comparison_json'].name}")
    if not is_pilot:
        print(f"   Comparativo CSV:  {fcfg['louvain_mst_comparison_csv'].name}")
    print("=" * 88)
    print("ETAPA 2 (LOUVAIN 100% PURO SOBRE A MST) CONCLUÍDA COM SUCESSO!")
    print("=" * 88 + "\n")

    return df_features, modularity_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Etapa 2: Executa o Algoritmo de Louvain 100% Puro sobre a Árvore Geradora Mínima (MST)."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=True,
        help="Executa no modo piloto 100x100 (padrão).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa no modo treino completo 1700x1700.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Parâmetro de resolução da modularidade no Louvain (padrão puro: 1.0).",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a analisar (padrão: speed_kmh).",
    )

    args = parser.parse_args()
    is_pilot_mode = not args.full
    run_mst_louvain(is_pilot=is_pilot_mode, resolution=args.resolution, feature=args.feature)
