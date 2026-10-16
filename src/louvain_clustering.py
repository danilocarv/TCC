"""
Módulo de Detecção de Comunidades via Algoritmo de Louvain sobre Grafo de Similaridade DTW.
Transforma a matriz de distâncias DTW de velocidade em um grafo de afinidade ponderado,
aplica o algoritmo de Louvain para maximização da Modularidade (Q), descobrindo a quantidade
ótima e natural de perfis de condução (sem necessidade de estipular k a priori), e gera o
mapa de calor reordenado por comunidades com fronteiras visuais.

Suporta:
- Modo Piloto: 100 x 100 janelas (validação preliminar e inspeção visual imediata);
- Modo Completo: 1.700 x 1.700 janelas (execução na base definitiva de treino).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import os
import sys
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
    DTW_PILOT_MATRIX_PARQUET,
    LOUVAIN_PILOT_CLUSTERS_CSV,
    LOUVAIN_PILOT_CLUSTERS_PARQUET,
    FIG_LOUVAIN_PILOT_HEATMAP,
    DTW_SPEED_MATRIX_NPY,
    DTW_SPEED_MATRIX_PARQUET,
    LOUVAIN_FULL_CLUSTERS_CSV,
    LOUVAIN_FULL_CLUSTERS_PARQUET,
    FIG_LOUVAIN_FULL_HEATMAP,
    LOUVAIN_KNN_NEIGHBORS,
    LOUVAIN_RANDOM_SEED,
    ensure_directories,
)


def build_dtw_affinity_graph(
    dist_matrix: np.ndarray,
    k_neighbors: int = LOUVAIN_KNN_NEIGHBORS,
    sigma: float | None = None,
) -> tuple[nx.Graph, float, np.ndarray]:
    """
    Constrói um grafo de afinidade ponderado a partir da matriz de distâncias DTW.
    1. Calcula a similaridade via kernel exponencial: S_ij = exp(-D_ij / sigma) para i != j;
    2. Conecta cada nó aos seus k vizinhos mais próximos (k-NN) para manter alta modularidade
       e eliminar ruído de fundo entre pares muito distantes.
    
    Retorna:
    - G: Grafo não direcionado ponderado (nx.Graph);
    - sigma: Largura de banda da similaridade;
    - S: Matriz densa de similaridade.
    """
    n = dist_matrix.shape[0]
    
    # Se sigma não fornecido, adota a mediana das distâncias fora da diagonal principal
    if sigma is None:
        off_diag = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        sigma = float(np.median(off_diag))
        if sigma <= 0.0:
            sigma = 1.0

    # Kernel exponencial de afinidade
    similarity_matrix = np.exp(-dist_matrix / sigma)
    np.fill_diagonal(similarity_matrix, 0.0)

    # Grafo não direcionado com poda k-NN
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        # Seleciona os k vizinhos com menor distância DTW (excluindo o próprio nó i)
        nearest_indices = np.argsort(dist_matrix[i])[1 : k_neighbors + 1]
        for j in nearest_indices:
            weight = float(similarity_matrix[i, j])
            # Se a aresta já existe de j para i, preserva o peso simétrico
            if G.has_edge(i, j):
                existing_w = G[i][j]["weight"]
                G[i][j]["weight"] = max(existing_w, weight)
            else:
                G.add_edge(i, j, weight=weight)

    return G, sigma, similarity_matrix


def detect_louvain_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = LOUVAIN_RANDOM_SEED,
) -> tuple[list[set[int]], float]:
    """
    Executa a partição de comunidades de Louvain maximizando a Modularidade (Q).
    """
    communities = nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    modularity_score = nx.community.modularity(
        G,
        communities,
        weight="weight",
        resolution=resolution,
    )
    return list(communities), float(modularity_score)


def sort_communities_by_speed(
    raw_communities: list[set[int]],
    df_features: pd.DataFrame,
) -> list[list[int]]:
    """
    Ordena as comunidades em ordem crescente de velocidade média e,
    dentro de cada comunidade, ordena as janelas também por velocidade.
    Isso produz um gradiente intuitivo e belo ao longo da diagonal do heatmap.
    """
    community_records = []
    for comm in raw_communities:
        nodes = list(comm)
        mean_v = float(df_features.iloc[nodes]["mean_speed_kmh"].mean())
        # Ordena as janelas internas por velocidade
        sorted_nodes = sorted(nodes, key=lambda idx: df_features.iloc[idx]["mean_speed_kmh"])
        community_records.append((mean_v, sorted_nodes))

    # Ordena as comunidades do menor para o maior perfil de velocidade média
    community_records.sort(key=lambda item: item[0])
    return [nodes for _, nodes in community_records]


def reorder_dtw_matrix(
    dist_matrix: np.ndarray,
    sorted_communities: list[list[int]],
) -> tuple[np.ndarray, list[int], list[int], dict[int, int]]:
    """
    Reordena linhas e colunas da matriz DTW de acordo com as comunidades do Louvain.
    Retorna:
    - reordered_dist: matriz reordenada;
    - ordered_indices: permutação dos nós (0 a N-1);
    - boundary_indices: pontos de corte para desenhar as linhas no heatmap;
    - node_to_comm: mapeamento nó original -> ID da comunidade (0, 1, ...).
    """
    ordered_indices = []
    boundary_indices = []
    node_to_comm = {}
    cum_count = 0

    for comm_id, nodes in enumerate(sorted_communities):
        for node in nodes:
            ordered_indices.append(node)
            node_to_comm[node] = comm_id
        cum_count += len(nodes)
        boundary_indices.append(cum_count)

    reordered_dist = dist_matrix[np.ix_(ordered_indices, ordered_indices)]
    return reordered_dist, ordered_indices, boundary_indices, node_to_comm


def plot_louvain_heatmap(
    matrix_reordered: np.ndarray,
    boundary_indices: list[int],
    community_labels: list[str],
    output_png: Path,
    title: str,
    subtitle: str | None = None,
):
    """
    Plota e salva o mapa de calor da matriz DTW reordenada pelas comunidades de Louvain,
    adicionando linhas delimitadoras vermelhas nas fronteiras de cada comunidade.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    k = matrix_reordered.shape[0]

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
    cbar.set_label("Distância DTW Acumulada sobre Velocidade (km/h)", fontsize=11, labelpad=10)

    # Desenha as linhas tracejadas delimitando as fronteiras das comunidades
    for b in boundary_indices[:-1]:
        ax.axvline(b - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axhline(b - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)

    # Posiciona os ticks nos centros das comunidades
    midpoints = []
    prev_b = 0
    for b in boundary_indices:
        midpoints.append((prev_b + b - 1) / 2.0)
        prev_b = b

    ax.set_xticks(midpoints)
    ax.set_xticklabels(community_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(midpoints)
    ax.set_yticklabels(community_labels, fontsize=10, fontweight="bold")

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    plt.title(full_title, fontsize=12, fontweight="bold", pad=14)
    plt.xlabel("Comunidades de Louvain (Janelas Reordenadas)", fontsize=11, labelpad=8)
    plt.ylabel("Comunidades de Louvain (Janelas Reordenadas)", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap de Louvain salvo em: {output_png.name} ({output_png})")


def print_community_report(
    df_summary: pd.DataFrame,
    modularity_q: float,
    sigma: float,
    k_neighbors: int,
    resolution: float,
):
    """
    Exibe um relatório detalhado e formatado no terminal com as características
    cinemáticas e espaciais de cada comunidade identificada pelo Louvain.
    """
    print("\n" + "=" * 92)
    print("           RELATÓRIO DE COMUNIDADES DE CONDUÇÃO (ALGORITMO DE LOUVAIN)")
    print("=" * 92)
    print(f"-> Parâmetros: k-NN = {k_neighbors} vizinhos | sigma = {sigma:.2f} km/h | Resolução = {resolution}")
    print(f"-> Modularidade da Rede (Q) = {modularity_q:.4f}  [Estrutura Significativa: Q > 0.30]")
    print(f"-> Quantidade de Comunidades Descobertas: {len(df_summary)}")
    print("-" * 92)
    
    header = (
        f"{'Comunidade':<12} | {'Qtd':<5} | {'%':<5} | {'v_méd (km/h)':<12} | "
        f"{'v_max (km/h)':<12} | {'a_méd (m/s²)':<12} | {'Fren/min':<9} | {'Acel/min':<9} | {'Tortuosidade':<12}"
    )
    print(header)
    print("-" * 92)

    for _, row in df_summary.iterrows():
        line = (
            f"{row['community_name']:<12} | "
            f"{int(row['count']):<5} | "
            f"{row['percentage']:>4.1f}% | "
            f"{row['mean_speed_kmh']:>12.2f} | "
            f"{row['max_speed_kmh']:>12.2f} | "
            f"{row['mean_pos_accel_ms2']:>12.2f} | "
            f"{row['hard_braking_rate_min']:>9.2f} | "
            f"{row['rapid_accel_rate_min']:>9.2f} | "
            f"{row['tortuosity_index']:>12.2f}"
        )
        print(line)

    print("=" * 92 + "\n")


def assign_profile_interpretation(mean_v: float, accel_rate: float, brake_rate: float) -> str:
    """Atribui uma interpretação heurística ao perfil de condução da comunidade."""
    if mean_v < 32.0:
        if brake_rate > 0.4:
            return "Urbano Congestionado"
        return "Urbano Fluido / Eco"
    elif mean_v < 55.0:
        if accel_rate > 0.6 or brake_rate > 0.8:
            return "Arterial Dinâmico"
        return "Arterial Moderado"
    else:
        if accel_rate > 0.8 or brake_rate > 1.0:
            return "Rodoviário Agressivo"
        return "Rodoviário / Cruzeiro"


def run_louvain_analysis(
    is_pilot: bool = True,
    k_neighbors: int = LOUVAIN_KNN_NEIGHBORS,
    resolution: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """
    Executa a análise completa de detecção de comunidades pelo Algoritmo de Louvain.
    """
    ensure_directories()
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 80)
    print(f"INICIANDO DETECÇÃO DE COMUNIDADES DE LOUVAIN SOBRE DTW - MODO {mode_str}")
    print("=" * 80)

    # 1. Carrega matriz DTW correspondente
    dtw_npy_path = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    if not dtw_npy_path.exists():
        raise FileNotFoundError(
            f"Matriz DTW não encontrada em {dtw_npy_path}. "
            f"Execute primeiro 'python src/dtw_analysis.py {'--pilot' if is_pilot else '--full'}'."
        )

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]
    print(f"-> Matriz DTW carregada com sucesso: {n_windows} x {n_windows} elementos.")

    # 2. Carrega features das janelas de treino para enriquecimento e interpretação
    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    # 3. Constrói grafo de afinidade ponderado com poda k-NN
    print(f"-> Construindo grafo de afinidade (k-NN = {k_neighbors})...")
    start_t = time.time()
    G, sigma, _ = build_dtw_affinity_graph(dist_matrix, k_neighbors=k_neighbors)
    print(f"   Grafo construído: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    print(f"   Largura de banda sigma calculada: {sigma:.2f} km/h.")

    # 4. Executa partição de Louvain
    print(f"-> Otimizando modularidade de rede via Louvain (resolução={resolution})...")
    raw_comms, modularity_q = detect_louvain_communities(G, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    elapsed_louvain = time.time() - start_t
    print(f"   Louvain concluído em {elapsed_louvain:.3f}s: {len(raw_comms)} comunidades identificadas.")
    print(f"   Modularidade Q = {modularity_q:.4f}")

    # 5. Ordena comunidades por velocidade média para visualização intuitiva
    sorted_comms = sort_communities_by_speed(raw_comms, df_features)

    # 6. Reordena a matriz DTW e obtém fronteiras
    reordered_dist, ordered_indices, boundary_indices, node_to_comm = reorder_dtw_matrix(
        dist_matrix, sorted_comms
    )

    # 7. Associa rótulos de comunidade ao dataframe de features
    df_features["louvain_community"] = [node_to_comm[i] for i in range(n_windows)]
    
    # Gera estatísticas resumidas por comunidade
    summary_list = []
    comm_labels_plot = []
    for c_id, nodes in enumerate(sorted_comms):
        sub = df_features[df_features["louvain_community"] == c_id]
        mean_v = float(sub["mean_speed_kmh"].mean())
        accel_r = float(sub["rapid_accel_rate_min"].mean())
        brake_r = float(sub["hard_braking_rate_min"].mean())
        interp = assign_profile_interpretation(mean_v, accel_r, brake_r)
        
        count = len(sub)
        pct = (count / n_windows) * 100.0
        
        summary_list.append({
            "community_id": c_id,
            "community_name": f"C{c_id} ({interp[:10]})",
            "interpretation": interp,
            "count": count,
            "percentage": pct,
            "mean_speed_kmh": mean_v,
            "max_speed_kmh": float(sub["max_speed_kmh"].mean()),
            "std_speed_kmh": float(sub["std_speed_kmh"].mean()),
            "mean_pos_accel_ms2": float(sub["mean_pos_accel_ms2"].mean()),
            "rapid_accel_rate_min": accel_r,
            "hard_braking_rate_min": brake_r,
            "high_jerk_rate_min": float(sub["high_jerk_rate_min"].mean()),
            "idle_ratio": float(sub["idle_ratio"].mean()),
            "tortuosity_index": float(sub["tortuosity_index"].mean()),
        })
        comm_labels_plot.append(f"C{c_id}\n(N={count})")

    df_summary = pd.DataFrame(summary_list)
    print_community_report(df_summary, modularity_q, sigma, k_neighbors, resolution)

    # 8. Gera mapa de calor reordenado com delimitadores visuais
    heatmap_path = FIG_LOUVAIN_PILOT_HEATMAP if is_pilot else FIG_LOUVAIN_FULL_HEATMAP
    plot_louvain_heatmap(
        matrix_reordered=reordered_dist,
        boundary_indices=boundary_indices,
        community_labels=comm_labels_plot,
        output_png=heatmap_path,
        title=f"Matriz de Similaridade DTW Reordenada por Comunidades de Louvain ({n_windows}x{n_windows})",
        subtitle=(
            f"Grafo de Afinidade k-NN (k={k_neighbors}) | sigma={sigma:.1f} km/h | "
            f"Modularidade Q = {modularity_q:.4f} ({len(sorted_comms)} Comunidades Naturais)"
        ),
    )

    # 9. Salva datasets de resultados em CSV e Parquet
    out_csv = LOUVAIN_PILOT_CLUSTERS_CSV if is_pilot else LOUVAIN_FULL_CLUSTERS_CSV
    out_parquet = LOUVAIN_PILOT_CLUSTERS_PARQUET if is_pilot else LOUVAIN_FULL_CLUSTERS_PARQUET

    # Enriquece dataframe original com informações de comunidade
    df_features["louvain_interpretation"] = df_features["louvain_community"].map(
        {row["community_id"]: row["interpretation"] for row in summary_list}
    )
    df_features.to_csv(out_csv, index=False, encoding="utf-8")
    df_features.to_parquet(out_parquet, index=False)
    print(f"-> Mapeamento de comunidades exportado:")
    print(f"   CSV:     {out_csv.name}")
    print(f"   Parquet: {out_parquet.name}")
    print("=" * 80)
    print("DETECÇÃO DE COMUNIDADES DE LOUVAIN CONCLUÍDA COM SUCESSO!")
    print("=" * 80 + "\n")

    return df_features, modularity_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa Detecção de Comunidades de Louvain sobre Grafo de Afinidade DTW."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=True,
        help="Executa o modo piloto com a matriz 100x100 (padrão).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa o modo completo com a matriz 1700x1700.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=LOUVAIN_KNN_NEIGHBORS,
        help=f"Número de vizinhos mais próximos para o grafo k-NN (padrão: {LOUVAIN_KNN_NEIGHBORS}).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Parâmetro de resolução da modularidade no Louvain (padrão: 1.0).",
    )

    args = parser.parse_args()
    is_pilot_mode = not args.full

    run_louvain_analysis(
        is_pilot=is_pilot_mode,
        k_neighbors=args.k_neighbors,
        resolution=args.resolution,
    )
