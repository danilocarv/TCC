"""
Módulo de Detecção de Comunidades via Algoritmo de Louvain sobre a Matriz DTW.
Opera de forma 100% pura e estritamente orientada à matriz de distâncias DTW:
1. Converte a matriz de distância DTW em afinidade contínua: W_ij = exp(-D_ij / sigma);
2. Constrói o grafo ponderado completo diretamente da matriz de afinidade;
3. Executa o algoritmo de Louvain (Blondel et al., 2008) maximizando a Modularidade (Q);
4. Reordena a matriz DTW apenas agrupando os nós da mesma comunidade (mantendo a ordem
   original dos índices dentro de cada comunidade, sem ordenação artificial por variáveis externas
   e sem atribuição heurística de rótulos).

Suporta:
- Modo Piloto: 100 x 100 janelas;
- Modo Completo: 1.700 x 1.700 janelas.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

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
    LOUVAIN_RANDOM_SEED,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)


def build_dtw_affinity_graph(
    dist_matrix: np.ndarray,
    sigma: float | None = None,
) -> tuple[nx.Graph, float, np.ndarray]:
    """
    Constrói o grafo de afinidade ponderado completo diretamente a partir da matriz DTW.
    Não utiliza k-NN, limiares de corte ou variáveis externas.

    A conversão distância -> afinidade (peso da aresta) utiliza o kernel exponencial:
    W_ij = exp(-D_ij / sigma) para i != j, e W_ii = 0,
    onde sigma é a mediana das distâncias fora da diagonal principal.
    """
    if sigma is None:
        off_diag = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        sigma = float(np.median(off_diag))
        if sigma <= 0.0:
            sigma = 1.0

    similarity_matrix = np.exp(-dist_matrix / sigma)
    np.fill_diagonal(similarity_matrix, 0.0)

    G = nx.from_numpy_array(similarity_matrix)
    return G, sigma, similarity_matrix


def detect_louvain_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = LOUVAIN_RANDOM_SEED,
) -> tuple[list[list[int]], float]:
    """
    Executa a partição de comunidades de Louvain maximizando a Modularidade (Q).
    Para garantir indexação determinística (C0, C1, ...) sem usar variáveis externas,
    as comunidades são indexadas apenas pelo menor índice de nó presente em cada grupo,
    e os nós internos mantêm estritamente sua ordem original de índice (0 a N-1).
    """
    raw_communities = nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    modularity_score = nx.community.modularity(
        G,
        raw_communities,
        weight="weight",
        resolution=resolution,
    )

    # Ordena os nós dentro de cada comunidade apenas pelo seu índice original (0..N-1)
    # e indexa as comunidades pelo menor índice de nó (100% cego a qualquer variável física)
    ordered_communities = [sorted(list(comm)) for comm in raw_communities]
    ordered_communities.sort(key=lambda nodes: nodes[0])

    return ordered_communities, float(modularity_score)


def reorder_dtw_matrix(
    dist_matrix: np.ndarray,
    communities: list[list[int]],
) -> tuple[np.ndarray, list[int], list[int], dict[int, int]]:
    """
    Reordena linhas e colunas da matriz DTW agrupando apenas os nós da mesma comunidade,
    preservando a ordem original de índice dentro de cada comunidade.
    """
    ordered_indices = []
    boundary_indices = []
    node_to_comm = {}
    cum_count = 0

    for comm_id, nodes in enumerate(communities):
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
    feature: str = "speed_kmh",
):
    """
    Plota e salva o mapa de calor da matriz DTW reordenada estritamente pelas
    comunidades do Louvain, sem ordenação secundária por variáveis externas.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)

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

    # Linhas divisórias tracejadas entre as comunidades
    for b in boundary_indices[:-1]:
        ax.axvline(b - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axhline(b - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)

    # Posiciona os ticks nos centros de cada bloco comunitário
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
    plt.xlabel("Comunidades de Louvain (Ordem Original Interna Preservada)", fontsize=11, labelpad=8)
    plt.ylabel("Comunidades de Louvain (Ordem Original Interna Preservada)", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap de Louvain salvo em: {output_png.name} ({output_png})")


def compute_pure_dtw_community_metrics(
    dist_matrix: np.ndarray,
    communities: list[list[int]],
) -> pd.DataFrame:
    """
    Calcula métricas puras da própria matriz DTW para cada comunidade:
    - Quantidade de janelas (nós);
    - Distância DTW média interna (intra-comunidade);
    - Distância DTW média externa (inter-comunidade).
    """
    n_total = dist_matrix.shape[0]
    all_indices = set(range(n_total))
    rows = []

    for comm_id, nodes in enumerate(communities):
        count = len(nodes)
        pct = (count / n_total) * 100.0

        # Distância média interna (entre membros da mesma comunidade)
        if count > 1:
            sub_mat = dist_matrix[np.ix_(nodes, nodes)]
            iu = np.triu_indices(count, k=1)
            intra_mean = float(sub_mat[iu].mean())
            intra_median = float(np.median(sub_mat[iu]))
        else:
            intra_mean = 0.0
            intra_median = 0.0

        # Distância média externa (para nós fora da comunidade)
        other_nodes = sorted(list(all_indices - set(nodes)))
        if other_nodes:
            inter_mat = dist_matrix[np.ix_(nodes, other_nodes)]
            inter_mean = float(inter_mat.mean())
        else:
            inter_mean = 0.0

        rows.append({
            "community_id": comm_id,
            "count": count,
            "percentage": pct,
            "intra_dtw_mean": intra_mean,
            "intra_dtw_median": intra_median,
            "inter_dtw_mean": inter_mean,
        })

    return pd.DataFrame(rows)


def print_pure_community_report(
    df_dtw_metrics: pd.DataFrame,
    df_features: pd.DataFrame,
    modularity_q: float,
    sigma: float,
    resolution: float,
):
    """
    Exibe o relatório em duas partes estritamente separadas:
    Parte 1: Métricas Topológicas Puras da Matriz DTW (sem variáveis externas);
    Parte 2: Estatísticas descritivas observadas a posteriori nas janelas de cada grupo
             (sem nenhum rótulo heurístico forçado).
    """
    print("\n" + "=" * 92)
    print("   PARTE 1: RESULTADO MATEMÁTICO PURO DO ALGORITMO DE LOUVAIN (MATRIZ DTW)")
    print("=" * 92)
    print(f"-> Grafo Completo de Afinidade (Sem k-NN) | sigma = {sigma:.2f} km/h | Resolução = {resolution}")
    print(f"-> Modularidade de Newman (Q) = {modularity_q:.4f}")
    print(f"-> Comunidades Detectadas: {len(df_dtw_metrics)}")
    print("-" * 92)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'Proporção':<10} | "
        f"{'DTW Interno Médio':<19} | {'DTW Externo Médio':<19} | {'Razão Int/Ext':<13}"
    )
    print("-" * 92)

    for _, row in df_dtw_metrics.iterrows():
        ratio = row["intra_dtw_mean"] / row["inter_dtw_mean"] if row["inter_dtw_mean"] > 0 else 0.0
        print(
            f"C{int(row['community_id']):<11} | "
            f"{int(row['count']):<8} | "
            f"{row['percentage']:>8.1f}%  | "
            f"{row['intra_dtw_mean']:>19.2f} | "
            f"{row['inter_dtw_mean']:>19.2f} | "
            f"{ratio:>13.3f}"
        )

    print("\n" + "=" * 92)
    print("   PARTE 2: MÉDIAS FÍSICAS OBSERVADAS A POSTERIORI POR COMUNIDADE (SEM RÓTULOS FORÇADOS)")
    print("=" * 92)
    print(
        f"{'Comunidade':<12} | {'v_méd (km/h)':<12} | {'std_v (km/h)':<12} | "
        f"{'v_max (km/h)':<12} | {'a_méd (m/s²)':<12} | {'Fren/min':<9} | {'Acel/min':<9}"
    )
    print("-" * 92)

    grouped = df_features.groupby("louvain_community")
    for comm_id, sub in grouped:
        print(
            f"C{int(comm_id):<11} | "
            f"{sub['mean_speed_kmh'].mean():>12.2f} | "
            f"{sub['std_speed_kmh'].mean():>12.2f} | "
            f"{sub['max_speed_kmh'].mean():>12.2f} | "
            f"{sub['mean_pos_accel_ms2'].mean():>12.2f} | "
            f"{sub['hard_braking_rate_min'].mean():>9.2f} | "
            f"{sub['rapid_accel_rate_min'].mean():>9.2f}"
        )
    print("=" * 92 + "\n")


def run_louvain_analysis(
    is_pilot: bool = True,
    resolution: float = 1.0,
    feature: str = "speed_kmh",
) -> tuple[pd.DataFrame, float]:
    """
    Executa a análise de Louvain 100% pura sobre a matriz DTW.
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 80)
    print(
        f"INICIANDO ALGORITMO DE LOUVAIN PURO SOBRE MATRIZ DTW ({fcfg['name_pt'].upper()}) - MODO {mode_str}"
    )
    print("=" * 80)

    # 1. Carrega exclusivamente a matriz DTW
    dtw_npy_path = fcfg["dtw_pilot_npy"] if is_pilot else fcfg["dtw_full_npy"]
    if not dtw_npy_path.exists():
        raise FileNotFoundError(
            f"Matriz DTW não encontrada em {dtw_npy_path}. "
            f"Execute primeiro 'python src/dtw_analysis.py --feature {feature} {'--pilot' if is_pilot else '--full'}'."
        )

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]
    print(f"-> Matriz DTW carregada ({fcfg['name_pt']}): {n_windows} x {n_windows} elementos.")

    # 2. Constrói grafo de afinidade ponderado completo (sem k-NN)
    print("-> Construindo grafo de afinidade completo (W_ij = exp(-D_ij / sigma))...")
    start_t = time.time()
    G, sigma, _ = build_dtw_affinity_graph(dist_matrix)
    print(f"   Grafo construído: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    print(f"   Largura de banda sigma (mediana DTW): {sigma:.2f} {fcfg['unit']}.")

    # 3. Executa partição de Louvain (100% cega a variáveis externas)
    print(f"-> Executando otimização de Modularidade de Louvain (resolução={resolution})...")
    communities, modularity_q = detect_louvain_communities(
        G, resolution=resolution, seed=LOUVAIN_RANDOM_SEED
    )
    elapsed_louvain = time.time() - start_t
    print(f"   Louvain concluído em {elapsed_louvain:.3f}s: {len(communities)} comunidades identificadas.")
    print(f"   Modularidade Q = {modularity_q:.4f}")

    # 4. Reordena a matriz DTW apenas agrupando os nós de cada comunidade
    reordered_dist, ordered_indices, boundary_indices, node_to_comm = reorder_dtw_matrix(
        dist_matrix, communities
    )

    # 5. Calcula métricas intrínsecas da matriz DTW por comunidade
    df_dtw_metrics = compute_pure_dtw_community_metrics(dist_matrix, communities)

    # 6. Gera o Heatmap 100% puro (sem ordenação por features externas)
    comm_labels_plot = [
        f"C{int(row['community_id'])}\n(N={int(row['count'])})"
        for _, row in df_dtw_metrics.iterrows()
    ]
    heatmap_path = fcfg["fig_louvain_pilot_heatmap"] if is_pilot else fcfg["fig_louvain_full_heatmap"]
    plot_louvain_heatmap(
        matrix_reordered=reordered_dist,
        boundary_indices=boundary_indices,
        community_labels=comm_labels_plot,
        output_png=heatmap_path,
        title=f"Matriz DTW ({fcfg['name_pt']}) Reordenada pelas Comunidades de Louvain ({n_windows}x{n_windows})",
        subtitle=(
            f"Louvain Puro (Sem k-NN, Sem Ordenação por Features) | "
            f"sigma={sigma:.2f} {fcfg['unit']} | Modularidade Q = {modularity_q:.4f}"
        ),
        feature=feature,
    )

    # 7. Anexa o ID numérico da comunidade (louvain_community) ao dataset para salvamento
    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    df_features["louvain_community"] = [node_to_comm[i] for i in range(n_windows)]

    print_pure_community_report(df_dtw_metrics, df_features, modularity_q, sigma, resolution)

    out_csv = fcfg["louvain_pilot_csv"] if is_pilot else fcfg["louvain_full_csv"]
    out_parquet = fcfg["louvain_pilot_parquet"] if is_pilot else fcfg["louvain_full_parquet"]

    df_features.to_csv(out_csv, index=False, encoding="utf-8")
    df_features.to_parquet(out_parquet, index=False)
    print("-> Mapeamento puro exportado (sem rótulos heurísticos):")
    print(f"   CSV:     {out_csv.name}")
    print(f"   Parquet: {out_parquet.name}")
    print("=" * 80)
    print("EXECUÇÃO DO LOUVAIN PURO CONCLUÍDA COM SUCESSO!")
    print("=" * 80 + "\n")

    return df_features, modularity_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa Detecção de Comunidades de Louvain Puro sobre a Matriz DTW."
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
        "--resolution",
        type=float,
        default=1.0,
        help="Parâmetro de resolução da modularidade no Louvain (padrão: 1.0).",
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

    run_louvain_analysis(
        is_pilot=is_pilot_mode,
        resolution=args.resolution,
        feature=args.feature,
    )
