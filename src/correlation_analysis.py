"""
Pipeline de Análise de Similaridade por Correlação de Pearson e Comunidades de Louvain.

Compara as séries temporais de condução dos 1.700 trajetos de treino usando
a Correlação de Pearson como linha de base clássica (baseline) frente ao DTW.

Funcionalidades:
1. Reamostra séries temporais (30s a 120s) em grid uniforme de 100 pontos normalizados;
2. Calcula matriz de correlação de Pearson completa (1.700 x 1.700) de forma vetorizada;
3. Constrói grafo completo ponderado pelas correlações positivas (sem filtros de percentil, sem MST);
4. Executa o Algoritmo de Louvain Puro para detecção não-supervisionada de comunidades;
5. Gera Heatmap reordenado em 300 DPI das comunidades;
6. Compara com os agrupamentos do DTW e do K-Means via Matriz de Cruzamento, ARI e NMI.

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
    clip_pos = fcfg.get("clip_positive", False)

    train_files = sorted(WINDOWS_TRAIN_PARQUET_DIR.glob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(
            f"Nenhum arquivo Parquet encontrado em {WINDOWS_TRAIN_PARQUET_DIR}."
        )

    if n_windows is not None:
        train_files = train_files[:n_windows]

    k = len(train_files)
    X = np.zeros((k, n_points), dtype=np.float64)
    file_stems = []
    target_grid = np.linspace(0.0, 1.0, n_points)

    for i, fpath in enumerate(train_files):
        df_win = pd.read_parquet(fpath, columns=[col_name])
        vals = df_win[col_name].to_numpy(dtype=np.float64)
        if clip_pos:
            vals = np.maximum(0.0, vals)

        n_orig = len(vals)
        if n_orig > 1:
            orig_grid = np.linspace(0.0, 1.0, n_orig)
            X[i, :] = np.interp(target_grid, orig_grid, vals)
        elif n_orig == 1:
            X[i, :] = vals[0]

        file_stems.append(fpath.stem)

    return X, file_stems


def compute_pearson_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de correlação de Pearson (K, K) de forma vetorizada via Z-score.
    R_ij = (Z_i . Z_j^T) / M
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)

    # Trata janelas com variância zero (ex: veículo totalmente parado a 0 km/h)
    valid_std = stds > 1e-8
    stds_safe = np.where(valid_std, stds, 1.0)

    Z = (X - means) / stds_safe
    Z = np.where(valid_std, Z, 0.0)

    m = X.shape[1]
    corr = (Z @ Z.T) / float(m)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return corr


def build_full_correlation_graph(corr_matrix: np.ndarray) -> tuple[nx.Graph, dict]:
    """
    Constrói o grafo completo não direcionado contendo todas as conexões positivas:
    W_ij = max(0, r_ij) para i != j.
    Nenhum corte de percentil e nenhum MST são aplicados; utiliza 100% da matriz positiva.
    """
    k = corr_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(k))

    rows, cols = np.triu_indices(k, k=1)
    weights = corr_matrix[rows, cols]

    # Mantém arestas com correlação estritamente positiva
    pos_mask = weights > 1e-6
    active_rows = rows[pos_mask]
    active_cols = cols[pos_mask]
    active_weights = weights[pos_mask]

    for u, v, w in zip(active_rows, active_cols, active_weights):
        G.add_edge(int(u), int(v), weight=float(w))

    num_possible = (k * (k - 1)) // 2
    retained_edges = G.number_of_edges()
    degrees = [int(d) for _, d in G.degree()]
    isolated_nodes = sum(1 for d in degrees if d == 0)

    info = {
        "n_nodes": k,
        "total_possible_pairs": num_possible,
        "positive_edges_retained": retained_edges,
        "pct_edges_retained": (retained_edges / num_possible) * 100.0 if num_possible > 0 else 0.0,
        "isolated_nodes_count": isolated_nodes,
        "is_connected": nx.is_connected(G) if retained_edges > 0 and isolated_nodes == 0 else False,
        "mean_degree": float(np.mean(degrees)),
        "mean_positive_correlation": float(np.mean(active_weights)) if len(active_weights) > 0 else 0.0,
    }

    return G, info


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
    Reordena as linhas e colunas da matriz para agrupar as comunidades.
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


def plot_raw_correlation_heatmap(
    corr_matrix: np.ndarray,
    output_png: Path,
    n_windows: int,
    feature: str = "speed_kmh",
) -> None:
    """
    Plota o Heatmap em 300 DPI da matriz de correlação de Pearson bruta
    (na ordem cronológica/original de índices 0 a N-1, sem reordenação por comunidades).
    Equivalente à figura 07_dtw_* gerada no pipeline do DTW.
    """
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
) -> None:
    """
    Plota o Heatmap em 300 DPI da matriz de correlação reordenada pelas comunidades de Louvain.
    """
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

    # Desenha linhas delimitadoras para comunidades reais (>= 2 nós)
    for idx, b in enumerate(boundary_indices[:-1]):
        if len(communities[idx]) >= 2:
            ax.axvline(b - 0.5, color="black", linestyle="--", linewidth=1.3, alpha=0.9)
            ax.axhline(b - 0.5, color="black", linestyle="--", linewidth=1.3, alpha=0.9)

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
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_yticks(midpoints)
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")

    plt.title(
        f"Matriz de Correlação de Pearson Reordenada — Louvain Completo ({n_windows}x{n_windows})\n"
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
    print(f"   Heatmap de Correlação salvo em: {output_png.name}")


def compute_intra_inter_correlation(
    corr_matrix: np.ndarray,
    communities: list[list[int]],
) -> dict[str, float]:
    """
    Calcula a correlação média dentro da mesma comunidade (intra) e
    entre comunidades diferentes (inter).
    """
    node_to_comm = {}
    for c_id, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = c_id

    k = corr_matrix.shape[0]
    intra_vals = []
    inter_vals = []

    for i in range(k):
        for j in range(i + 1, k):
            r = corr_matrix[i, j]
            if node_to_comm.get(i) == node_to_comm.get(j):
                intra_vals.append(r)
            else:
                inter_vals.append(r)

    mean_intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    mean_inter = float(np.mean(inter_vals)) if inter_vals else 0.0

    return {
        "mean_intra_correlation": mean_intra,
        "mean_inter_correlation": mean_inter,
        "diff_intra_inter": mean_intra - mean_inter,
    }


def compare_with_dtw_and_kmeans(
    df_features: pd.DataFrame,
    feature: str = "speed_kmh",
    is_pilot: bool = False,
) -> dict:
    """
    Cruza as comunidades de Louvain obtidas pela Correlação contra:
    1. As comunidades de Louvain obtidas pelo DTW (mesma feature).
    2. Os clusters obtidos pelo K-Means Baseline (driving_features).
    """
    fcfg = get_feature_paths(feature)
    results = {}

    # 1. Comparação com K-Means
    km_labeled_path = PROCESSED_FEATURES_DIR / "driving_windows_train_labeled.parquet"
    if km_labeled_path.exists() and not is_pilot:
        df_km = pd.read_parquet(km_labeled_path)
        if "cluster_label" in df_km.columns:
            ct_km = pd.crosstab(
                df_features["corr_louvain_community"],
                df_km["cluster_label"],
                margins=True,
            )
            ari_km = float(adjusted_rand_score(df_km["cluster_label"], df_features["corr_louvain_community"]))
            nmi_km = float(normalized_mutual_info_score(df_km["cluster_label"], df_features["corr_louvain_community"]))
            results["kmeans_comparison"] = {
                "ari": ari_km,
                "nmi": nmi_km,
                "crosstab": ct_km.to_dict(),
            }
            print("\n   --- Cruzamento: Louvain (Correlação) vs. K-Means ---")
            print(ct_km)
            print(f"   ARI: {ari_km:.4f} | NMI: {nmi_km:.4f}")

    # 2. Comparação com DTW Louvain
    dtw_clusters_path = fcfg["louvain_pilot_parquet"] if is_pilot else fcfg["louvain_full_parquet"]
    if dtw_clusters_path.exists():
        df_dtw = pd.read_parquet(dtw_clusters_path)
        if "louvain_community" in df_dtw.columns:
            ct_dtw = pd.crosstab(
                df_features["corr_louvain_community"],
                df_dtw["louvain_community"],
                margins=True,
            )
            ari_dtw = float(adjusted_rand_score(df_dtw["louvain_community"], df_features["corr_louvain_community"]))
            nmi_dtw = float(normalized_mutual_info_score(df_dtw["louvain_community"], df_features["corr_louvain_community"]))
            results["dtw_comparison"] = {
                "ari": ari_dtw,
                "nmi": nmi_dtw,
                "crosstab": ct_dtw.to_dict(),
            }
            print(f"\n   --- Cruzamento: Louvain (Correlação) vs. Louvain (DTW {feature}) ---")
            print(ct_dtw)
            print(f"   ARI: {ari_dtw:.4f} | NMI: {nmi_dtw:.4f}")

    return results


def run_correlation_pipeline(
    feature: str = "speed_kmh",
    is_pilot: bool = False,
    n_points: int = 100,
    resolution: float = 1.0,
) -> dict:
    """
    Executa o pipeline completo de Correlação de Pearson + Louvain para a feature informada.
    """
    fcfg = get_feature_paths(feature)
    tag = "Piloto (100x100)" if is_pilot else "Completo (1700x1700)"
    n_windows = DTW_PILOT_SIZE if is_pilot else None

    print(f"\n{'='*75}")
    print(f"PIPELINE DE CORRELAÇÃO DE PEARSON — {fcfg['name_pt'].upper()} [{tag}]")
    print(f"{'='*75}")

    # 1. Carrega e reamostra séries
    t0 = time.time()
    X, file_stems = load_and_resample_series(feature=feature, n_windows=n_windows, n_points=n_points)
    k = X.shape[0]
    print(f"-> {k:,d} séries carregadas e reamostradas para grid uniforme de {n_points} pontos.")

    # 2. Calcula matriz de correlação de Pearson completa
    corr_matrix = compute_pearson_correlation_matrix(X)
    t_corr = time.time() - t0
    print(f"-> Matriz de Correlação de Pearson ({k}x{k}) calculada em {t_corr:.3f} s.")

    # Salva matriz de correlação em NPY e Parquet
    npy_path = fcfg["corr_pilot_npy"] if is_pilot else fcfg["corr_full_npy"]
    parquet_path = fcfg["corr_pilot_parquet"] if is_pilot else fcfg["corr_full_parquet"]
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, corr_matrix)
    pd.DataFrame(corr_matrix, index=file_stems, columns=file_stems).to_parquet(parquet_path)
    print(f"-> Matriz salva em: {parquet_path.name}")

    # 2.1 Plota Heatmap da Matriz Bruta de Correlação (Ordem Original dos Trajetos)
    raw_heatmap_path = fcfg["fig_corr_pilot_heatmap"] if is_pilot else fcfg["fig_corr_full_heatmap"]
    plot_raw_correlation_heatmap(
        corr_matrix=corr_matrix,
        output_png=raw_heatmap_path,
        n_windows=k,
        feature=feature,
    )

    # 3. Constrói grafo ponderado completo (W_ij = max(0, r_ij))
    G, topo_info = build_full_correlation_graph(corr_matrix)
    print(f"-> Grafo completo construído: {G.number_of_edges():,d} arestas positivas "
          f"({topo_info['pct_edges_retained']:.2f}% dos pares). "
          f"Nós órfãos: {topo_info['isolated_nodes_count']}.")

    # 4. Detecção de Comunidades Louvain Puro
    t_louvain_0 = time.time()
    communities, modularity_q = detect_louvain_communities(G, resolution=resolution, seed=LOUVAIN_RANDOM_SEED)
    t_louvain = time.time() - t_louvain_0
    n_comm = len(communities)
    n_active = sum(1 for c in communities if len(c) > 1)
    print(f"-> Louvain convergiu em {t_louvain:.3f} s: {n_active} comunidades principais "
          f"({n_comm} totais), Modularidade Q = {modularity_q:.4f}.")

    # 5. Reordena e plota Heatmap Reordenado pelas Comunidades de Louvain
    corr_reordered, ordered_indices, boundary_indices, node_to_comm = reorder_matrix_by_communities(
        corr_matrix, communities
    )
    louvain_heatmap_path = (
        fcfg["fig_corr_louvain_pilot_heatmap"] if is_pilot else fcfg["fig_corr_louvain_full_heatmap"]
    )
    plot_correlation_heatmap(
        corr_reordered=corr_reordered,
        boundary_indices=boundary_indices,
        communities=communities,
        output_png=louvain_heatmap_path,
        n_windows=k,
        modularity_q=modularity_q,
        feature=feature,
    )

    # 6. Salva clusters atribuídos
    train_feat_path = PROCESSED_FEATURES_DIR / "driving_windows_train_features.parquet"
    if train_feat_path.exists() and not is_pilot:
        df_feat = pd.read_parquet(train_feat_path).iloc[:k].copy()
    else:
        df_feat = pd.DataFrame({"window_stem": file_stems})

    df_feat["corr_louvain_community"] = [node_to_comm[i] for i in range(k)]
    df_feat["corr_degree"] = [int(G.degree(i)) for i in range(k)]

    clusters_csv = fcfg["corr_louvain_pilot_csv"] if is_pilot else fcfg["corr_louvain_full_csv"]
    clusters_parquet = fcfg["corr_louvain_pilot_parquet"] if is_pilot else fcfg["corr_louvain_full_parquet"]
    df_feat.to_csv(clusters_csv, index=False)
    df_feat.to_parquet(clusters_parquet, index=False)
    print(f"-> Rótulos de comunidades salvos em: {clusters_parquet.name}")

    # 7. Análise Intra vs Inter correlação
    intra_inter = compute_intra_inter_correlation(corr_matrix, communities)
    print(f"-> Correlação Média Intra-Comunidade: {intra_inter['mean_intra_correlation']:.4f}")
    print(f"-> Correlação Média Inter-Comunidades: {intra_inter['mean_inter_correlation']:.4f}")
    print(f"-> Diferença (Intra - Inter): {intra_inter['diff_intra_inter']:.4f}")

    # 8. Cruzamento com DTW e K-Means
    comp_results = compare_with_dtw_and_kmeans(df_feat, feature=feature, is_pilot=is_pilot)

    # 9. Salva relatório em JSON
    summary_report = {
        "feature": feature,
        "is_pilot": is_pilot,
        "n_windows": k,
        "topology": topo_info,
        "louvain": {
            "resolution": resolution,
            "modularity_q": modularity_q,
            "n_total_communities": n_comm,
            "n_active_communities": n_active,
            "community_sizes": [len(c) for c in communities],
        },
        "intra_inter_correlation": intra_inter,
        "comparisons": comp_results,
    }

    json_path = fcfg["corr_vs_dtw_comparison_json"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"-> Relatório de síntese salvo em: {json_path.name}")

    return summary_report


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Correlação de Pearson e Louvain para Trajetos de Treino."
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
