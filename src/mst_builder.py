"""
Módulo da Etapa 1: Construção e Análise Pura da Árvore Geradora Mínima (MST) sobre a Matriz DTW.

Opera de forma 100% pura e estritamente orientada à Matriz de Distâncias DTW:
1. Carrega a matriz DTW completa (100x100 no piloto ou 1.700x1.700 no treino);
2. Constrói o grafo completo considerando exclusivamente pares distintos (i < j),
   garantindo que a diagonal principal (D_ii = 0) jamais forme auto-laços (self-loops),
   e atribuindo cada valor D_ij diretamente como o peso da aresta;
3. Executa o Algoritmo de Kruskal (nx.minimum_spanning_tree) para extrair a
   Árvore Geradora Mínima (MST) com exatamente N nós e N - 1 arestas de menor soma DTW;
4. Exporta as arestas da árvore (.csv e .parquet), a matriz da árvore (.npy),
   as métricas estruturais (.json) e a visualização gráfica da MST pura (300 DPI).

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
    DTW_PILOT_SIZE,
    DTW_PILOT_MATRIX_NPY,
    DTW_SPEED_MATRIX_NPY,
    MST_PILOT_EDGES_CSV,
    MST_PILOT_EDGES_PARQUET,
    MST_PILOT_MATRIX_NPY,
    FIG_MST_PURE_PILOT,
    MST_FULL_EDGES_CSV,
    MST_FULL_EDGES_PARQUET,
    MST_FULL_MATRIX_NPY,
    FIG_MST_PURE_FULL,
    MST_STRUCTURAL_METRICS_JSON,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)


def build_complete_dtw_graph(dist_matrix: np.ndarray) -> nx.Graph:
    """
    Constrói o grafo completo não-direcionado a partir da matriz DTW inteira,
    iterando estritamente sobre o triângulo superior (i < j).

    Garante matematicamente:
    - Todos os pares distintos (i, j) entram com peso 'weight' = D_ij;
    - A diagonal principal (i == j, onde D_ii = 0) NUNCA é adicionada (zero auto-laços).
    """
    n = dist_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    rows, cols = np.triu_indices(n, k=1)
    weights = dist_matrix[rows, cols]

    edges = [
        (int(u), int(v), {"weight": float(w), "dtw_distance": float(w)})
        for u, v, w in zip(rows, cols, weights)
    ]
    G.add_edges_from(edges)

    assert nx.number_of_selfloops(G) == 0, "O grafo não pode conter auto-laços da diagonal!"
    return G


def extract_minimum_spanning_tree(
    G_full: nx.Graph,
    dist_matrix: np.ndarray,
    window_files: list[str],
) -> tuple[nx.Graph, pd.DataFrame, np.ndarray, dict]:
    """
    Executa o Algoritmo de Kruskal sobre o grafo completo DTW para extrair a
    Árvore Geradora Mínima (MST) pura.

    Retorna:
    - T_mst: objeto nx.Graph representando a árvore geradora mínima;
    - df_edges: DataFrame das N - 1 arestas ordenado por distância DTW crescente (ordem de Kruskal);
    - mst_matrix: matriz simétrica N x N contendo apenas as arestas da MST;
    - metrics: dicionário com métricas matemáticas e topológicas da árvore.
    """
    n = dist_matrix.shape[0]
    start_t = time.time()

    # Algoritmo de Kruskal sobre o peso 'weight' (distância DTW pura D_ij)
    T_mst = nx.minimum_spanning_tree(G_full, weight="weight", algorithm="kruskal")
    elapsed_s = time.time() - start_t

    # Auditoria matemática estrita da árvore
    assert T_mst.number_of_nodes() == n, f"Esperado {n} nós, obtido {T_mst.number_of_nodes()}"
    assert T_mst.number_of_edges() == n - 1, (
        f"Uma árvore com {n} nós deve ter exatamente {n - 1} arestas, obtido {T_mst.number_of_edges()}"
    )
    assert nx.number_of_selfloops(T_mst) == 0, "A MST não pode conter auto-laços da diagonal!"
    assert nx.is_connected(T_mst), "A MST deve ser 100% conexa (sem nós órfãos)!"
    assert nx.is_tree(T_mst), "O subgrafo gerado deve ser estritamente uma árvore (acíclico)!"

    # Constrói tabela de arestas e matriz de adjacência da MST
    mst_matrix = np.zeros((n, n), dtype=np.float64)
    edge_rows = []

    for u, v, data in T_mst.edges(data=True):
        u_idx, v_idx = (min(u, v), max(u, v))
        dtw_val = float(data["weight"])
        mst_matrix[u_idx, v_idx] = dtw_val
        mst_matrix[v_idx, u_idx] = dtw_val
        edge_rows.append({
            "node_u": int(u_idx),
            "node_v": int(v_idx),
            "window_u": window_files[u_idx],
            "window_v": window_files[v_idx],
            "dtw_distance": dtw_val,
        })

    # Ordena na sequência exata de seleção do Algoritmo de Kruskal (menor DTW primeiro)
    df_edges = pd.DataFrame(edge_rows).sort_values(
        by=["dtw_distance", "node_u", "node_v"], ascending=[True, True, True]
    ).reset_index(drop=True)
    df_edges.insert(0, "kruskal_step", range(1, len(df_edges) + 1))

    # Estatísticas comparativas: Grafo Completo vs. MST
    off_diag_weights = dist_matrix[np.triu_indices(n, k=1)]
    mst_weights = df_edges["dtw_distance"].to_numpy(dtype=np.float64)

    degrees = dict(T_mst.degree())
    degree_values = np.array(list(degrees.values()), dtype=np.int64)
    leaf_nodes = int(np.sum(degree_values == 1))
    max_deg = int(degree_values.max())

    # Identifica os top 5 nós mais conectados (Hubs da Árvore)
    top_hubs = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))[:5]
    hubs_info = [
        {"node_id": int(node), "window_file": window_files[node], "degree": int(deg)}
        for node, deg in top_hubs
    ]

    # Diâmetro da árvore (em número de saltos/arestas)
    tree_diameter_hops = int(nx.diameter(T_mst))

    metrics = {
        "n_nodes": int(n),
        "full_graph_edges": int(len(off_diag_weights)),
        "mst_edges": int(len(mst_weights)),
        "edge_reduction_pct": float((1.0 - len(mst_weights) / len(off_diag_weights)) * 100.0),
        "is_connected": bool(nx.is_connected(T_mst)),
        "is_tree": bool(nx.is_tree(T_mst)),
        "self_loops": int(nx.number_of_selfloops(T_mst)),
        "kruskal_execution_time_s": float(elapsed_s),
        "full_graph_dtw_sum": float(off_diag_weights.sum()),
        "mst_dtw_total_cost": float(mst_weights.sum()),
        "full_graph_dtw_mean": float(off_diag_weights.mean()),
        "full_graph_dtw_median": float(np.median(off_diag_weights)),
        "mst_dtw_min": float(mst_weights.min()),
        "mst_dtw_mean": float(mst_weights.mean()),
        "mst_dtw_median": float(np.median(mst_weights)),
        "mst_dtw_std": float(mst_weights.std()),
        "mst_dtw_max": float(mst_weights.max()),
        "mean_degree": float(degree_values.mean()),
        "max_degree": max_deg,
        "leaf_nodes_count": leaf_nodes,
        "leaf_nodes_pct": float((leaf_nodes / n) * 100.0),
        "tree_diameter_hops": tree_diameter_hops,
        "top_hubs": hubs_info,
    }

    return T_mst, df_edges, mst_matrix, metrics


def plot_pure_mst_figure(
    T_mst: nx.Graph,
    dist_matrix: np.ndarray,
    df_features: pd.DataFrame,
    metrics: dict,
    output_png: Path,
    is_pilot: bool,
    feature: str = "speed_kmh",
):
    """
    Plota um painel duplo de alta resolução (300 DPI) da Etapa 1 (MST Pura):
    - Painel Esquerdo: Topologia da Árvore Geradora Mínima (Kruskal), com nós coloridos
      a posteriori pela velocidade média (mean_speed_kmh) e tamanho proporcional ao grau na árvore;
    - Painel Direito: Comparação da distribuição de distâncias DTW de todas as arestas da matriz
      versus as N - 1 arestas selecionadas pelo Algoritmo de Kruskal.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    n = T_mst.number_of_nodes()

    fig, (ax_tree, ax_hist) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

    # Layout da árvore: spring_layout com semente fixa para 1700 nós (rápido e bem distribuído)
    # ou kamada_kawai_layout no piloto 100x100
    if is_pilot:
        pos = nx.kamada_kawai_layout(T_mst, weight=None)
    else:
        pos = nx.spring_layout(T_mst, k=1.8 / np.sqrt(n), iterations=120, seed=42, weight=None)

    fcfg = get_feature_paths(feature)
    if feature == "accel_ms2":
        grad_values = df_features["mean_pos_accel_ms2"].to_numpy(dtype=np.float64)
        cbar_title = "Aceleração Média Positiva (m/s²) [Observada a Posteriori]"
    elif feature in ["power_kw", "power_pos_kw"]:
        grad_values = df_features["mean_power_kw"].to_numpy(dtype=np.float64)
        cbar_title = "Potência Média da Janela (kW) [Observada a Posteriori]"
    else:
        grad_values = df_features["mean_speed_kmh"].to_numpy(dtype=np.float64)
        cbar_title = "Velocidade Média da Janela (km/h) [Observada a Posteriori]" 
    degrees = np.array([T_mst.degree(i) for i in range(n)], dtype=np.float64)

    #Tamanho dos nós destaca sutilmente os hubs da árvore
    if is_pilot:
        node_sizes = 45.0 + 28.0 * degrees
        edge_alpha = 0.45
        edge_width = 1.0
    else:
        node_sizes = 12.0 + 8.0 * degrees
        edge_alpha = 0.25
        edge_width = 0.55

    nx.draw_networkx_edges(
        T_mst,
        pos=pos,
        ax=ax_tree,
        edge_color="#555555",
        width=edge_width,
        alpha=edge_alpha,
    )

    nodes_scatter = nx.draw_networkx_nodes(
        T_mst,
        pos=pos,
        ax=ax_tree,
        node_size=node_sizes,
        node_color=grad_values,
        cmap="plasma",
        alpha=0.92,
         linewidths=0.3,
        edgecolors="black",
    )

    cbar = plt.colorbar(nodes_scatter, ax=ax_tree, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_title, fontsize=10)

    ax_tree.set_title(
        f"Topologia da Árvore Geradora Mínima Pura (Algoritmo de Kruskal)\n"
        f"{n} Nós Conectados | {metrics['mst_edges']} Arestas (0 Ciclos, 0 Auto-laços)",
        fontsize=11,
        fontweight="bold",
    )
    ax_tree.axis("off")

    # Painel Direito: Histograma das distâncias DTW (Matriz Inteira vs. Arestas da MST)
    off_diag = dist_matrix[np.triu_indices(n, k=1)]
    mst_weights = np.array([d["weight"] for _, _, d in T_mst.edges(data=True)], dtype=np.float64)

    ax_hist.hist(
        off_diag,
        bins=45,
        density=True,
        alpha=0.45,
        color="#6c757d",
        edgecolor="black",
        linewidth=0.4,
        label=f"Matriz DTW Inteira ({len(off_diag):,d} pares)\nMédia = {off_diag.mean():.1f} {fcfg['unit']}",
    )
    ax_hist.hist(
        mst_weights,
        bins=35,
        density=True,
        alpha=0.75,
        color="#0077b6",
        edgecolor="black",
        linewidth=0.5,
        label=f"Arestas Selecionadas na MST ({len(mst_weights):,d} arestas)\nMédia = {mst_weights.mean():.1f} {fcfg['unit']}",
    )

    ax_hist.axvline(
        mst_weights.mean(),
        color="#03045e",
        linestyle="--",
        linewidth=1.8,
        label=f"Média MST ({mst_weights.mean():.1f} {fcfg['unit']})",
    )
    ax_hist.axvline(
        off_diag.mean(),
        color="#343a40",
        linestyle=":",
        linewidth=1.8,
        label=f"Média Matriz Completa ({off_diag.mean():.1f} {fcfg['unit']})",
    )

    ax_hist.set_title(
        "Seletividade do Algoritmo de Kruskal sobre os Pesos DTW\n"
        "A MST retém exclusivamente as transições de menor distância DTW",
        fontsize=11,
        fontweight="bold",
    )
    ax_hist.set_xlabel(f"Distância DTW Acumulada sobre {fcfg['name_pt']} ({fcfg['unit']})", fontsize=10)
    ax_hist.set_ylabel("Densidade de Probabilidade", fontsize=10)
    ax_hist.legend(loc="upper right", fontsize=9, frameon=True)
    ax_hist.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        f"ETAPA 1: Árvore Geradora Mínima (MST) Extraída da Matriz DTW ({n}x{n})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Figura da MST Pura salva em: {output_png.name} ({output_png})")


def print_mst_report(metrics: dict, df_edges: pd.DataFrame):
    """Imprime relatório detalhado da Árvore Geradora Mínima construída na Etapa 1."""
    print("\n" + "=" * 88)
    print("   RELATÓRIO MATEMÁTICO E ESTRUTURAL DA ÁRVORE GERADORA MÍNIMA (ETAPA 1 - KRUSKAL)")
    print("=" * 88)
    print(f"-> Vértices (Janelas de Condução):      {metrics['n_nodes']:,d} nós (100% conectados)")
    print(f"-> Arestas na Matriz DTW Inteira:       {metrics['full_graph_edges']:,d} arestas")
    print(f"-> Arestas na MST (N - 1):              {metrics['mst_edges']:,d} arestas")
    print(f"-> Auto-laços da Diagonal (i == j):     {metrics['self_loops']} (diagonal zero ignorada)")
    print(f"-> Redução de Arestas Redundantes:      {metrics['edge_reduction_pct']:.2f}%")
    print(f"-> Grafo Conexo / Árvore Acíclica:      is_connected={metrics['is_connected']} | is_tree={metrics['is_tree']}")
    print("-" * 88)
    print(f"-> Custo DTW Total (Grafo Completo):    {metrics['full_graph_dtw_sum']:,.2f} km/h")
    print(f"-> Custo DTW Total Mínimo (MST):        {metrics['mst_dtw_total_cost']:,.2f} km/h")
    print(f"-> Distância DTW Média (Grafo Completo):{metrics['full_graph_dtw_mean']:,.2f} km/h")
    print(f"-> Distância DTW Média (Arestas da MST):{metrics['mst_dtw_mean']:,.2f} km/h (Mediana: {metrics['mst_dtw_median']:.2f})")
    print(f"-> Faixa de Pesos DTW na MST:           Mín = {metrics['mst_dtw_min']:.2f} | Máx = {metrics['mst_dtw_max']:.2f} km/h")
    print("-" * 88)
    print(f"-> Grau Médio na Árvore:                {metrics['mean_degree']:.3f} (Grau Máximo = {metrics['max_degree']})")
    print(f"-> Nós Folha (Grau = 1):                {metrics['leaf_nodes_count']} ({metrics['leaf_nodes_pct']:.1f}% dos nós)")
    print(f"-> Diâmetro Topológico da Árvore:       {metrics['tree_diameter_hops']} saltos")
    print("-> Top 5 Nós Centrais (Hubs da MST):")
    for h in metrics["top_hubs"]:
        print(f"     • Nó {h['node_id']:4d} ({h['window_file']}): conectado a {h['degree']} janelas vizinhas")
    print("-" * 88)
    print("-> Primeiras 5 arestas adicionadas pelo Algoritmo de Kruskal (menores distâncias DTW):")
    for _, row in df_edges.head(5).iterrows():
        print(
            f"     Passo {int(row['kruskal_step']):4d}: "
            f"Nó {int(row['node_u']):4d} ({row['window_u']}) <-> "
            f"Nó {int(row['node_v']):4d} ({row['window_v']}) | "
            f"DTW = {row['dtw_distance']:.2f} km/h"
        )
    print("=" * 88 + "\n")


def update_structural_metrics_json(
    mode_key: str,
    metrics: dict,
    json_path: Path = MST_STRUCTURAL_METRICS_JSON,
):
    """Atualiza o arquivo JSON consolidado com as métricas estruturais da MST."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload[mode_key] = metrics
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_mst_builder(
    is_pilot: bool = True,
    feature: str = "speed_kmh",
) -> tuple[nx.Graph, pd.DataFrame, dict]:
    """Executa a Etapa 1 completa: construção, auditoria e salvamento da MST pura."""
    ensure_directories()
    fcfg = get_feature_paths(feature)
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 88)
    print(
        f"INICIANDO ETAPA 1: CONSTRUÇÃO DA ÁRVORE GERADORA MÍNIMA (MST PURA) [{fcfg['name_pt'].upper()}] - MODO {mode_str}"
    )
    print("=" * 88)

    dtw_npy_path = fcfg["dtw_pilot_npy"] if is_pilot else fcfg["dtw_full_npy"]
    if not dtw_npy_path.exists():
        raise FileNotFoundError(
            f"Matriz DTW não encontrada em {dtw_npy_path}. "
            f"Execute primeiro 'python src/dtw_analysis.py --feature {feature} {'--pilot' if is_pilot else '--full'}'."
        )

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]
    print(f"-> Matriz DTW carregada ({fcfg['name_pt']}): {n_windows} x {n_windows} elementos.")

    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    window_files = df_features["window_file"].astype(str).tolist()

    print("-> Construindo grafo completo a partir da matriz DTW inteira (excluindo diagonal i == j)...")
    G_full = build_complete_dtw_graph(dist_matrix)
    print(
        f"   Grafo de entrada pronto: {G_full.number_of_nodes()} nós e "
        f"{G_full.number_of_edges():,d} arestas ponderadas por D_ij."
    )

    print("-> Executando Algoritmo de Kruskal para extrair a Árvore Geradora Mínima (MST)...")
    T_mst, df_edges, mst_matrix, metrics = extract_minimum_spanning_tree(
        G_full, dist_matrix, window_files
    )
    print(
        f"   MST extraída em {metrics['kruskal_execution_time_s']:.3f}s: "
        f"{T_mst.number_of_nodes()} nós e {T_mst.number_of_edges()} arestas essenciais."
    )

    print_mst_report(metrics, df_edges)

    # Salva os artefatos da Etapa 1
    edges_csv = fcfg["mst_pilot_edges_csv"] if is_pilot else fcfg["mst_full_edges_csv"]
    edges_parquet = fcfg["mst_pilot_edges_parquet"] if is_pilot else fcfg["mst_full_edges_parquet"]
    mst_npy = fcfg["mst_pilot_matrix_npy"] if is_pilot else fcfg["mst_full_matrix_npy"]
    fig_path = fcfg["fig_mst_pure_pilot"] if is_pilot else fcfg["fig_mst_pure_full"]
    json_path = fcfg["mst_structural_metrics_json"]

    df_edges.to_csv(edges_csv, index=False, encoding="utf-8")
    df_edges.to_parquet(edges_parquet, index=False)
    np.save(mst_npy, mst_matrix)

    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"
    update_structural_metrics_json(mode_key, metrics, json_path=json_path)

    plot_pure_mst_figure(
        T_mst=T_mst,
        dist_matrix=dist_matrix,
        df_features=df_features,
        metrics=metrics,
        output_png=fig_path,
        is_pilot=is_pilot,
        feature=feature,
    )

    print("-> Artefatos da Etapa 1 (MST Pura) salvos com sucesso:")
    print(f"   Arestas CSV:     {edges_csv.name}")
    print(f"   Arestas Parquet: {edges_parquet.name}")
    print(f"   Matriz MST NPY:  {mst_npy.name}")
    print(f"   Métricas JSON:   {json_path.name}")
    print("=" * 88)
    print("ETAPA 1 (ÁRVORE GERADORA MÍNIMA PURA) CONCLUÍDA COM SUCESSO!")
    print("=" * 88 + "\n")

    return T_mst, df_edges, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Etapa 1: Constrói a Árvore Geradora Mínima (MST) pura sobre a Matriz DTW."
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
        "--feature",
        type=str,
        default="speed_kmh",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a analisar (padrão: speed_kmh).",
    )

    args = parser.parse_args()
    is_pilot_mode = not args.full
    run_mst_builder(is_pilot=is_pilot_mode, feature=args.feature)
