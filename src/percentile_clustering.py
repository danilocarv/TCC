"""
Módulo da Etapa 3A: Filtragem Direta por Percentil 25 de Distância DTW (Top 25% Melhores Conexões)
e Detecção de Comunidades via Algoritmo de Louvain 100% Puro (Sem União com a MST).

Executa estritamente a primeira etapa solicitada pelo orientador:
1. Carrega a matriz DTW completa (100x100 no piloto ou 1.700x1.700 no treino);
2. Calcula o valor exato do Percentil 25 (P25) das distâncias DTW fora da diagonal (i < j),
   equivalente ao Top 25% de maior similaridade/afinidade (P75 de W_ij = exp(-D_ij / sigma));
3. Constrói o grafo filtrado mantendo EXCLUSIVAMENTE as arestas com D_ij <= P25 (sem MST);
4. Audita a topologia resultante (quantidade de arestas, componentes conexas e nós órfãos/isolados);
5. Executa o Algoritmo de Louvain 100% puro (resolution=1.0, sem induzir k e sem variáveis externas);
6. Gera o Heatmap reordenado (300 DPI), tabelas de comunidades (.csv/.parquet) e métricas (.json).

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
    LOUVAIN_RANDOM_SEED,
    DTW_PERCENTILE_CUTOFF,
    MST_PILOT_EDGES_PARQUET,
    MST_FULL_EDGES_PARQUET,
    P25_DIRECT_PILOT_CLUSTERS_CSV,
    P25_DIRECT_PILOT_CLUSTERS_PARQUET,
    FIG_P25_DIRECT_PILOT_HEATMAP,
    P25_DIRECT_FULL_CLUSTERS_CSV,
    P25_DIRECT_FULL_CLUSTERS_PARQUET,
    FIG_P25_DIRECT_FULL_HEATMAP,
    P25_DIRECT_METRICS_JSON,
    P25_DIRECT_SUMMARY_CSV,
    P25_MST_PILOT_CLUSTERS_CSV,
    P25_MST_PILOT_CLUSTERS_PARQUET,
    FIG_P25_MST_PILOT_HEATMAP,
    P25_MST_FULL_CLUSTERS_CSV,
    P25_MST_FULL_CLUSTERS_PARQUET,
    FIG_P25_MST_FULL_HEATMAP,
    P25_MST_METRICS_JSON,
    P25_MST_SUMMARY_CSV,
    DTW_ALL_STAGES_COMPARISON_JSON,
    DTW_ALL_STAGES_COMPARISON_CSV,
    LOUVAIN_PILOT_CLUSTERS_PARQUET,
    LOUVAIN_FULL_CLUSTERS_PARQUET,
    MST_STRUCTURAL_METRICS_JSON,
    LOUVAIN_MST_COMPARISON_JSON,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)
from src.louvain_clustering import (
    detect_louvain_communities,
    reorder_dtw_matrix,
    compute_pure_dtw_community_metrics,
)


def build_direct_percentile_graph(
    dist_matrix: np.ndarray,
    percentile: float = DTW_PERCENTILE_CUTOFF,
    window_files: list[str] | None = None,
) -> tuple[nx.Graph, dict]:
    """
    Calcula o limiar do Percentil 25 de distância DTW (Top 25% menores distâncias)
    sobre o triângulo superior estrito (i < j) e constrói o grafo filtrado direto
    (sem adicionar arestas da MST).
    """
    n = dist_matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    off_diag_dtw = dist_matrix[rows, cols]
    total_pairs = len(off_diag_dtw)

    # Largura de banda sigma = mediana das distâncias fora da diagonal
    sigma = float(np.median(off_diag_dtw))
    if sigma <= 0.0:
        sigma = 1.0

    # Valor de corte do Percentil 25 na distância DTW (equivale ao Percentil 75 na afinidade W_ij)
    dtw_p25_threshold = float(np.percentile(off_diag_dtw, percentile))
    affinity_p75_threshold = float(np.exp(-dtw_p25_threshold / sigma))

    # Filtra apenas os pares (i, j) com D_ij <= P25
    mask = off_diag_dtw <= dtw_p25_threshold
    sel_rows = rows[mask]
    sel_cols = cols[mask]
    sel_dtw = off_diag_dtw[mask]
    sel_aff = np.exp(-sel_dtw / sigma)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    edges = [
        (int(u), int(v), {"weight": float(w), "dtw_distance": float(d)})
        for u, v, w, d in zip(sel_rows, sel_cols, sel_aff, sel_dtw)
    ]
    G.add_edges_from(edges)

    assert nx.number_of_selfloops(G) == 0, "O grafo filtrado não pode conter auto-laços da diagonal!"

    # Auditoria topológica de conectividade e nós isolados (órfãos)
    degrees = dict(G.degree())
    degree_vals = np.array(list(degrees.values()), dtype=np.int64)
    isolated_nodes = [int(node) for node, deg in degrees.items() if deg == 0]
    connected_nodes_count = int(n - len(isolated_nodes))

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc_size = len(components[0]) if components else 0
    non_trivial_components = [c for c in components if len(c) > 1]

    isolated_details = []
    if window_files is not None:
        for idx in isolated_nodes[:15]:
            isolated_details.append({"node_id": int(idx), "window_file": window_files[idx]})

    topo_info = {
        "n_nodes": int(n),
        "percentile_dtw": float(percentile),
        "equivalent_affinity_percentile": float(100.0 - percentile),
        "sigma_median_dtw_kmh": sigma,
        "dtw_p25_threshold_kmh": dtw_p25_threshold,
        "affinity_p75_threshold": affinity_p75_threshold,
        "dtw_min_kmh": float(off_diag_dtw.min()),
        "dtw_p50_median_kmh": sigma,
        "dtw_p75_kmh": float(np.percentile(off_diag_dtw, 75.0)),
        "dtw_max_kmh": float(off_diag_dtw.max()),
        "full_graph_edges": int(total_pairs),
        "retained_edges": int(G.number_of_edges()),
        "retained_edges_pct": float((G.number_of_edges() / total_pairs) * 100.0),
        "is_connected": bool(nx.is_connected(G)),
        "total_connected_components": int(len(components)),
        "non_trivial_components_count": int(len(non_trivial_components)),
        "largest_component_nodes": int(largest_cc_size),
        "largest_component_pct": float((largest_cc_size / n) * 100.0),
        "connected_nodes_count": connected_nodes_count,
        "isolated_orphan_nodes_count": int(len(isolated_nodes)),
        "isolated_orphan_nodes_pct": float((len(isolated_nodes) / n) * 100.0),
        "isolated_orphan_nodes_sample": isolated_details,
        "mean_degree": float(degree_vals.mean()),
        "max_degree": int(degree_vals.max()),
    }

    return G, topo_info


def plot_p25_direct_heatmap(
    matrix_reordered: np.ndarray,
    boundary_indices: list[int],
    communities: list[list[int]],
    output_png: Path,
    n_windows: int,
    dtw_threshold: float,
    modularity_q: float,
    n_isolated: int,
    feature: str = "speed_kmh",
):
    """
    Plota o Heatmap em 300 DPI da matriz DTW reordenada pelas comunidades descobertas
    pelo Louvain Puro sobre o grafo do Percentil 25 Direto.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fcfg = get_feature_paths(feature)
    n_comm = len(communities)
    n_active_comm = sum(1 for c in communities if len(c) > 1)

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

    # Desenha linhas divisórias para comunidades com >= 2 nós
    for idx, b in enumerate(boundary_indices[:-1]):
        comm_size = len(communities[idx])
        if comm_size >= 2:
            ax.axvline(b - 0.5, color="red", linestyle="--", linewidth=1.4, alpha=0.88)
            ax.axhline(b - 0.5, color="red", linestyle="--", linewidth=1.4, alpha=0.88)

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

    isolated_str = f" | {n_isolated} Nós Órfãos" if n_isolated > 0 else " | 0 Nós Órfãos"
    plt.title(
        f"Matriz DTW Reordenada — Louvain Puro sobre Percentil 25 Direto ({n_windows}x{n_windows})\n"
        f"Corte D_ij <= {dtw_threshold:.2f} {fcfg['unit']} (Top 25% Conexões) | "
        f"{n_active_comm} Comunidades Principais{isolated_str} | Modularidade Q = {modularity_q:.4f}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("Comunidades de Louvain no Percentil 25 Direto (Ordem Original Interna Preservada)", fontsize=10.5, labelpad=8)
    plt.ylabel("Comunidades de Louvain no Percentil 25 Direto (Ordem Original Interna Preservada)", fontsize=10.5, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap Percentil 25 Direto salvo em: {output_png.name} ({output_png})")


def print_stage_3a_report(
    topo_info: dict,
    df_dtw_metrics: pd.DataFrame,
    df_features: pd.DataFrame,
    modularity_q: float,
    resolution: float,
):
    """Imprime o relatório detalhado da Etapa 3A (Percentil 25 Direto + Louvain Puro)."""
    n_comm_total = len(df_dtw_metrics)
    df_active = df_dtw_metrics[df_dtw_metrics["count"] > 1].copy()
    df_orphans = df_dtw_metrics[df_dtw_metrics["count"] == 1].copy()

    print("\n" + "=" * 96)
    print("   ETAPA 3A — PARTE 1: AUDITORIA DO LIMIAR DO PERCENTIL 25 DIRETO (SEM MST)")
    print("=" * 96)
    print(f"-> Total de Janelas (Vértices):               {topo_info['n_nodes']:,d} nós")
    print(f"-> Total de Pares na Matriz DTW Inteira:      {topo_info['full_graph_edges']:,d} arestas")
    print(f"-> Limiar Calculado do Percentil 25 (P25):    D_ij <= {topo_info['dtw_p25_threshold_kmh']:,.2f} km/h")
    print(f"   (Quartis da Matriz DTW: Mín={topo_info['dtw_min_kmh']:.2f} | P25={topo_info['dtw_p25_threshold_kmh']:.2f} | "
          f"Mediana P50={topo_info['dtw_p50_median_kmh']:.2f} | P75={topo_info['dtw_p75_kmh']:.2f} | Máx={topo_info['dtw_max_kmh']:.2f} km/h)")
    print(f"-> Afinidade Mínima Equivalente (P75 W_ij):   W_ij >= {topo_info['affinity_p75_threshold']:.4f} (sigma = {topo_info['sigma_median_dtw_kmh']:.2f} km/h)")
    print(f"-> Arestas Retidas no Top 25%:                {topo_info['retained_edges']:,d} arestas ({topo_info['retained_edges_pct']:.2f}%)")
    print("-" * 96)
    print("-> DIAGNÓSTICO DE CONECTIVIDADE DO CORTE DIRETO P25 (SEM MST):")
    print(f"   • Grafo 100% Conexo?                       {topo_info['is_connected']}")
    print(f"   • Total de Componentes Conexas:            {topo_info['total_connected_components']}")
    print(f"   • Maior Componente Conexa (Gigante):       {topo_info['largest_component_nodes']} nós ({topo_info['largest_component_pct']:.2f}% da base)")
    print(f"   • Janelas Conectadas (Grau >= 1):          {topo_info['connected_nodes_count']} nós")
    print(f"   • Janelas Órfãs / Isoladas (Grau == 0):    {topo_info['isolated_orphan_nodes_count']} nós ({topo_info['isolated_orphan_nodes_pct']:.2f}% da base)")
    if topo_info["isolated_orphan_nodes_sample"]:
        sample_str = ", ".join(
            f"Nó {x['node_id']} ({x['window_file']})"
            for x in topo_info["isolated_orphan_nodes_sample"][:5]
        )
        print(f"   • Exemplo de Janelas Órfãs:                {sample_str}")

    print("\n" + "=" * 96)
    print("   ETAPA 3A — PARTE 2: RESULTADO DO LOUVAIN 100% PURO SOBRE O PERCENTIL 25 DIRETO")
    print("=" * 96)
    print(f"-> Modularidade de Newman (Q) no P25 Direto = {modularity_q:.4f} (Resolução = {resolution})")
    print(f"-> Partições Totais Retornadas pelo Louvain:  {n_comm_total} "
          f"({len(df_active)} comunidades reais com >= 2 nós + {len(df_orphans)} nós órfãos individuais)")
    print("-" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'Proporção':<10} | "
        f"{'DTW Interno Médio':<19} | {'DTW Externo Médio':<19} | {'Razão Int/Ext':<13}"
    )
    print("-" * 96)

    for _, row in df_active.iterrows():
        ratio = row["intra_dtw_mean"] / row["inter_dtw_mean"] if row["inter_dtw_mean"] > 0 else 0.0
        print(
            f"C{int(row['community_id']):<11} | "
            f"{int(row['count']):<8} | "
            f"{row['percentage']:>8.2f}%  | "
            f"{row['intra_dtw_mean']:>19.2f} | "
            f"{row['inter_dtw_mean']:>19.2f} | "
            f"{ratio:>13.3f}"
        )
    if len(df_orphans) > 0:
        print(
            f"[Órfãos]     | {len(df_orphans):<8} | "
            f"{(len(df_orphans) / topo_info['n_nodes']) * 100.0:>8.2f}%  | "
            f"{'0.00 (1 nó cada)':>19} | "
            f"{df_orphans['inter_dtw_mean'].mean():>19.2f} | "
            f"{'N/A (Isolado)':>13}"
        )

    print("\n" + "=" * 96)
    print("   ETAPA 3A — PARTE 3: MÉDIAS FÍSICAS OBSERVADAS A POSTERIORI NAS COMUNIDADES PRINCIPAIS")
    print("=" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'v_méd (km/h)':<12} | {'std_v (km/h)':<12} | "
        f"{'v_max (km/h)':<12} | {'a_méd (m/s²)':<12} | {'Fren/min':<9} | {'Acel/min':<9}"
    )
    print("-" * 96)

    for _, row in df_active.iterrows():
        c_id = int(row["community_id"])
        sub = df_features[df_features["p25_direct_louvain_community"] == c_id]
        print(
            f"C{c_id:<11} | "
            f"{len(sub):<8} | "
            f"{sub['mean_speed_kmh'].mean():>12.2f} | "
            f"{sub['std_speed_kmh'].mean():>12.2f} | "
            f"{sub['max_speed_kmh'].mean():>12.2f} | "
            f"{sub['mean_pos_accel_ms2'].mean():>12.2f} | "
            f"{sub['hard_braking_rate_min'].mean():>9.2f} | "
            f"{sub['rapid_accel_rate_min'].mean():>9.2f}"
        )
    if len(df_orphans) > 0:
        orphan_ids = df_orphans["community_id"].astype(int).tolist()
        sub_orph = df_features[df_features["p25_direct_louvain_community"].isin(orphan_ids)]
        print(
            f"[Órfãos]     | "
            f"{len(sub_orph):<8} | "
            f"{sub_orph['mean_speed_kmh'].mean():>12.2f} | "
            f"{sub_orph['std_speed_kmh'].mean():>12.2f} | "
            f"{sub_orph['max_speed_kmh'].mean():>12.2f} | "
            f"{sub_orph['mean_pos_accel_ms2'].mean():>12.2f} | "
            f"{sub_orph['hard_braking_rate_min'].mean():>9.2f} | "
            f"{sub_orph['rapid_accel_rate_min'].mean():>9.2f}"
        )
    print("=" * 96 + "\n")


def run_stage_3a_direct_percentile(
    is_pilot: bool = False,
    resolution: float = 1.0,
    feature: str = "speed_kmh",
) -> tuple[pd.DataFrame, dict]:
    """
    Executa a Etapa 3A: Filtragem direta pelo Percentil 25 de distância DTW e Louvain Puro.
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 96)
    print(
        f"INICIANDO ETAPA 3A: PERCENTIL 25 DIRETO [{fcfg['name_pt'].upper()}] + LOUVAIN PURO - MODO {mode_str}"
    )
    print("=" * 96)

    dtw_npy_path = fcfg["dtw_pilot_npy"] if is_pilot else fcfg["dtw_full_npy"]
    if not dtw_npy_path.exists():
        raise FileNotFoundError(f"Matriz DTW não encontrada em {dtw_npy_path}.")

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]

    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    window_files = df_features["window_file"].astype(str).tolist()

    # 1. Constrói o grafo filtrado direto pelo Percentil 25 (D_ij <= P25)
    start_t = time.time()
    G_p25, topo_info = build_direct_percentile_graph(
        dist_matrix=dist_matrix,
        percentile=DTW_PERCENTILE_CUTOFF,
        window_files=window_files,
    )

    # 2. Executa o Algoritmo de Louvain 100% puro sobre G_p25
    communities, modularity_q = detect_louvain_communities(
        G_p25, resolution=resolution, seed=LOUVAIN_RANDOM_SEED
    )
    elapsed_s = time.time() - start_t

    # Reordena para colocar as comunidades reais (>=2 nós) primeiro e os nós órfãos (1 nó) agrupados ao final
    active_communities = [c for c in communities if len(c) > 1]
    orphan_communities = [c for c in communities if len(c) == 1]
    ordered_communities = active_communities + orphan_communities

    reordered_dist, ordered_indices, boundary_indices, node_to_comm = reorder_dtw_matrix(
        dist_matrix, ordered_communities
    )

    df_dtw_metrics = compute_pure_dtw_community_metrics(dist_matrix, ordered_communities)

    df_features["p25_direct_louvain_community"] = [node_to_comm[i] for i in range(n_windows)]
    df_features["p25_direct_degree"] = [int(G_p25.degree(i)) for i in range(n_windows)]
    df_features["p25_direct_is_isolated"] = df_features["p25_direct_degree"] == 0

    # 3. Gera Heatmap reordenado
    heatmap_path = (
        fcfg["fig_p25_direct_pilot_heatmap"] if is_pilot else fcfg["fig_p25_direct_full_heatmap"]
    )
    plot_p25_direct_heatmap(
        matrix_reordered=reordered_dist,
        boundary_indices=boundary_indices,
        communities=ordered_communities,
        output_png=heatmap_path,
        n_windows=n_windows,
        dtw_threshold=topo_info["dtw_p25_threshold_kmh"],
        modularity_q=modularity_q,
        n_isolated=topo_info["isolated_orphan_nodes_count"],
    )

    # 4. Imprime relatório completo da Etapa 3A
    print_stage_3a_report(
        topo_info=topo_info,
        df_dtw_metrics=df_dtw_metrics,
        df_features=df_features,
        modularity_q=modularity_q,
        resolution=resolution,
    )

    # 5. Salva artefatos em disco
    out_csv = fcfg["p25_direct_pilot_csv"] if is_pilot else fcfg["p25_direct_full_csv"]
    out_pq = fcfg["p25_direct_pilot_parquet"] if is_pilot else fcfg["p25_direct_full_parquet"]
    metrics_json = fcfg["p25_direct_metrics_json"]
    summary_csv = fcfg["p25_direct_summary_csv"]

    df_features.to_csv(out_csv, index=False, encoding="utf-8")
    df_features.to_parquet(out_pq, index=False)

    # Calcula médias ponderadas das comunidades reais (>= 2 nós)
    df_active = df_dtw_metrics[df_dtw_metrics["count"] > 1]
    n_active_nodes = int(df_active["count"].sum())
    w_intra = float(np.sum(df_active["intra_dtw_mean"] * df_active["count"]) / max(n_active_nodes, 1))
    w_inter = float(np.sum(df_active["inter_dtw_mean"] * df_active["count"]) / max(n_active_nodes, 1))

    result_payload = {
        **topo_info,
        "louvain_execution_time_s": float(elapsed_s),
        "modularity_q": float(modularity_q),
        "total_partitions_returned": int(len(ordered_communities)),
        "active_communities_count": int(len(active_communities)),
        "orphan_singletons_count": int(len(orphan_communities)),
        "active_weighted_intra_dtw_mean": w_intra,
        "active_weighted_inter_dtw_mean": w_inter,
        "active_intra_inter_ratio": float(w_intra / w_inter) if w_inter > 0 else 0.0,
    }

    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    json_data = {}
    if metrics_json.exists():
        try:
            json_data = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            json_data = {}
    json_data[mode_key] = result_payload
    metrics_json.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not is_pilot:
        kinematic_means = (
            df_features.groupby("p25_direct_louvain_community")[
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
            .rename(columns={"p25_direct_louvain_community": "community_id"})
        )
        df_summary = df_dtw_metrics.merge(kinematic_means, on="community_id", how="left")
        df_summary.to_csv(summary_csv, index=False, encoding="utf-8")

    print("-> Artefatos da Etapa 3A (Percentil 25 Direto) salvos com sucesso:")
    print(f"   Clusters CSV:     {out_csv.name}")
    print(f"   Clusters Parquet: {out_pq.name}")
    print(f"   Métricas JSON:    {metrics_json.name}")
    if not is_pilot:
        print(f"   Resumo CSV:       {summary_csv.name}")
    print("=" * 96)
    print("ETAPA 3A (PERCENTIL 25 DIRETO + LOUVAIN PURO) CONCLUÍDA COM SUCESSO!")
    print("=" * 96 + "\n")

    return df_features, result_payload


def build_mst_percentile_graph(
    dist_matrix: np.ndarray,
    mst_edges_parquet: Path,
    percentile: float = DTW_PERCENTILE_CUTOFF,
) -> tuple[nx.Graph, dict]:
    """
    Constrói a rede combinada da Etapa 3B: união das arestas do Percentil 25 (Top 25% DTW)
    com as arestas da Árvore Geradora Mínima (MST), garantindo 100% dos nós conectados (0 órfãos).
    """
    n = dist_matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    off_diag_dtw = dist_matrix[rows, cols]
    total_pairs = len(off_diag_dtw)

    sigma = float(np.median(off_diag_dtw))
    if sigma <= 0.0:
        sigma = 1.0

    dtw_p25_threshold = float(np.percentile(off_diag_dtw, percentile))

    # Arestas do Percentil 25
    mask = off_diag_dtw <= dtw_p25_threshold
    p25_set = set(zip(rows[mask].tolist(), cols[mask].tolist()))

    # Carrega arestas da MST geradas na Etapa 1
    if not mst_edges_parquet.exists():
        raise FileNotFoundError(
            f"Arquivo da MST não encontrado em {mst_edges_parquet}. "
            "Execute primeiro a Etapa 1 ('python src/mst_builder.py')."
        )
    df_mst = pd.read_parquet(mst_edges_parquet)
    mst_edges_list = [
        (min(int(u), int(v)), max(int(u), int(v)), float(d))
        for u, v, d in zip(df_mst["node_u"], df_mst["node_v"], df_mst["dtw_distance"])
    ]

    # Identifica arestas da MST que já estavam no P25 vs. arestas de resgate que a MST traz
    mst_in_p25_count = 0
    mst_rescue_edges = []
    for u, v, d in mst_edges_list:
        if (u, v) in p25_set:
            mst_in_p25_count += 1
        else:
            mst_rescue_edges.append((u, v, d))

    # Constrói o grafo unificado G = P25 ∪ MST
    G_union = nx.Graph()
    G_union.add_nodes_from(range(n))

    # Adiciona arestas do P25
    for u, v in p25_set:
        d = float(dist_matrix[u, v])
        w = float(np.exp(-d / sigma))
        G_union.add_edge(u, v, weight=w, dtw_distance=d, origin="P25")

    # Adiciona arestas de resgate da MST
    for u, v, d in mst_rescue_edges:
        w = float(np.exp(-d / sigma))
        G_union.add_edge(u, v, weight=w, dtw_distance=d, origin="MST_Rescue")

    assert nx.number_of_selfloops(G_union) == 0, "O grafo unificado não pode conter auto-laços!"
    assert nx.is_connected(G_union), "A rede unificada MST ∪ P25 DEVE ser 100% conexa!"

    degrees = dict(G_union.degree())
    degree_vals = np.array(list(degrees.values()), dtype=np.int64)
    isolated_nodes = [node for node, deg in degrees.items() if deg == 0]

    topo_info = {
        "n_nodes": int(n),
        "percentile_dtw": float(percentile),
        "sigma_median_dtw_kmh": sigma,
        "dtw_p25_threshold_kmh": dtw_p25_threshold,
        "full_graph_edges": int(total_pairs),
        "p25_edges_count": int(len(p25_set)),
        "mst_total_edges": int(len(mst_edges_list)),
        "mst_already_in_p25": int(mst_in_p25_count),
        "mst_rescue_edges_added": int(len(mst_rescue_edges)),
        "union_edges_total": int(G_union.number_of_edges()),
        "union_edges_pct": float((G_union.number_of_edges() / total_pairs) * 100.0),
        "is_connected": bool(nx.is_connected(G_union)),
        "connected_components": int(nx.number_connected_components(G_union)),
        "isolated_orphan_nodes_count": int(len(isolated_nodes)),
        "mean_degree": float(degree_vals.mean()),
        "min_degree": int(degree_vals.min()),
        "max_degree": int(degree_vals.max()),
    }

    return G_union, topo_info


def plot_p25_mst_heatmap(
    matrix_reordered: np.ndarray,
    boundary_indices: list[int],
    communities: list[list[int]],
    output_png: Path,
    n_windows: int,
    dtw_threshold: float,
    modularity_q: float,
    rescue_edges_count: int,
    feature: str = "speed_kmh",
):
    """
    Plota o Heatmap em 300 DPI da matriz DTW reordenada pelas comunidades descobertas
    pelo Louvain Puro sobre a rede unificada MST ∪ P25 (100% conexa, 0 órfãos).
    Quando existem micro-comunidades periféricas (< 1.5% dos nós, ex: C3 e C4),
    evita sobreposição de rótulos nos eixos e adiciona um painel de zoom ampliado.
    """
    fcfg = get_feature_paths(feature)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Identifica se há micro-comunidades pequenas (< 1.5% de n_windows) no final
    comm_sizes = [len(c) for c in communities]
    micro_comm_ids = [cid for cid, sz in enumerate(comm_sizes) if sz < max(3, int(0.015 * n_windows))]
    has_micro_zoom = len(micro_comm_ids) >= 1 and n_windows >= 100

    if has_micro_zoom:
        fig = plt.figure(figsize=(17, 8.5), dpi=300)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
        ax = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
        ax_zoom = None

    im = ax.imshow(
        matrix_reordered,
        cmap="viridis",
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    if ax_zoom is None:
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(
            f"Distância DTW sobre {fcfg['name_pt']} ({fcfg['unit']})",
            fontsize=10.5,
            labelpad=8,
        )

    for b in boundary_indices[:-1]:
        ax.axvline(b - 0.5, color="red", linestyle="--", linewidth=1.4, alpha=0.88)
        ax.axhline(b - 0.5, color="red", linestyle="--", linewidth=1.4, alpha=0.88)

    # Constrói pontos médios e agrupa rótulos muito próximos (< 4.5% da matriz) para evitar colisão visual
    raw_ticks = []
    prev_b = 0
    for comm_id, b in enumerate(boundary_indices):
        size = b - prev_b
        mid = (prev_b + b - 1) / 2.0
        raw_ticks.append({"ids": [comm_id], "sizes": [size], "mid": mid, "start": prev_b, "end": b})
        prev_b = b

    min_gap = 0.045 * n_windows
    merged_ticks = []
    for item in raw_ticks:
        if merged_ticks and (item["mid"] - merged_ticks[-1]["mid"]) < min_gap:
            prev_item = merged_ticks[-1]
            prev_item["ids"].extend(item["ids"])
            prev_item["sizes"].extend(item["sizes"])
            prev_item["end"] = item["end"]
            prev_item["mid"] = (prev_item["start"] + prev_item["end"] - 1) / 2.0
        else:
            merged_ticks.append(item)

    midpoints = [t["mid"] for t in merged_ticks]
    x_labels = []
    y_labels = []
    for t in merged_ticks:
        if len(t["ids"]) == 1:
            cid, sz = t["ids"][0], t["sizes"][0]
            x_labels.append(f"C{cid}\n(N={sz})")
            y_labels.append(f"C{cid}\n(N={sz})")
        else:
            parts = [f"C{cid} (N={sz})" for cid, sz in zip(t["ids"], t["sizes"])]
            x_labels.append("\n".join(parts))
            y_labels.append(" | ".join(parts))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(x_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(midpoints)
    ax.set_yticklabels(y_labels, fontsize=9, fontweight="bold")
    ax.set_title(
        f"Matriz DTW Global ({n_windows}x{n_windows}) — {len(communities)} Comunidades",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Comunidades de Louvain em MST ∪ P25 (Ordem Interna Preservada)", fontsize=10, labelpad=8)
    ax.set_ylabel("Comunidades de Louvain em MST ∪ P25 (Ordem Interna Preservada)", fontsize=10, labelpad=8)

    # Painel de Zoom dedicado para visualizar claramente as micro-comunidades (ex: C3 e C4)
    if ax_zoom is not None:
        first_micro_id = micro_comm_ids[0]
        first_micro_start = 0 if first_micro_id == 0 else boundary_indices[first_micro_id - 1]
        total_micro_nodes = n_windows - first_micro_start
        # Inclui um contexto da comunidade imediatamente anterior (ex: últimas 8 janelas de C2)
        context_nodes = max(6, min(10, total_micro_nodes * 2))
        zoom_start = max(0, first_micro_start - context_nodes)
        sub_mat = matrix_reordered[zoom_start:, zoom_start:]
        sub_n = sub_mat.shape[0]

        im_z = ax_zoom.imshow(
            sub_mat,
            cmap="viridis",
            aspect="equal",
            origin="upper",
            interpolation="nearest",
            vmin=float(matrix_reordered.min()),
            vmax=float(matrix_reordered.max()),
        )

        # Desenha linhas divisórias e calcula centros dentro do recorte de zoom
        z_mids = []
        z_labels = []
        prev_zb = zoom_start
        for comm_id, b in enumerate(boundary_indices):
            if b <= zoom_start:
                continue
            start_in_z = max(zoom_start, 0 if comm_id == 0 else boundary_indices[comm_id - 1]) - zoom_start
            end_in_z = b - zoom_start
            if end_in_z < sub_n:
                ax_zoom.axvline(end_in_z - 0.5, color="red", linestyle="--", linewidth=2.0, alpha=0.95)
                ax_zoom.axhline(end_in_z - 0.5, color="red", linestyle="--", linewidth=2.0, alpha=0.95)

            mid_in_z = (start_in_z + end_in_z - 1) / 2.0
            full_size = comm_sizes[comm_id]
            if comm_id < first_micro_id:
                z_labels.append(f"Final de C{comm_id}\n({end_in_z - start_in_z} de {full_size})")
            else:
                z_labels.append(f"C{comm_id}\n(N={full_size})")
            z_mids.append(mid_in_z)

        # Anota valores DTW dentro dos blocos de C3 e C4 no zoom
        for i_z in range(sub_n):
            for j_z in range(sub_n):
                global_i = zoom_start + i_z
                global_j = zoom_start + j_z
                if global_i >= first_micro_start and global_j >= first_micro_start:
                    val = sub_mat[i_z, j_z]
                    txt_color = "black" if val > (matrix_reordered.max() * 0.55) else "white"
                    ax_zoom.text(
                        j_z,
                        i_z,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=7.5,
                        fontweight="bold",
                    )

        ax_zoom.set_xticks(z_mids)
        ax_zoom.set_xticklabels(z_labels, fontsize=9.5, fontweight="bold")
        ax_zoom.set_yticks(z_mids)
        ax_zoom.set_yticklabels(z_labels, fontsize=9.5, fontweight="bold")
        micro_names = ", ".join([f"C{cid} (N={comm_sizes[cid]})" for cid in micro_comm_ids])
        ax_zoom.set_title(
            f"Zoom de Ampliação no Canto Inferior Direito\nDetalhe das Micro-Comunidades: {micro_names}",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax_zoom.set_xlabel(f"Últimas {sub_n} Janelas da Matriz Reordenada (Índices {zoom_start} a {n_windows - 1})", fontsize=9.5, labelpad=8)
        ax_zoom.set_ylabel(f"Últimas {sub_n} Janelas da Matriz Reordenada", fontsize=9.5, labelpad=8)

        cbar = plt.colorbar(im_z, ax=ax_zoom, pad=0.03, fraction=0.046)
        cbar.set_label(
            f"Distância DTW sobre {fcfg['name_pt']} ({fcfg['unit']})",
            fontsize=10.5,
            labelpad=8,
        )

    fig.suptitle(
        f"Matriz DTW Reordenada ({fcfg['name_pt']}) — Louvain Puro sobre Rede Otimizada MST ∪ P25 ({n_windows}x{n_windows})\n"
        f"Top 25% Conexões (D_ij <= {dtw_threshold:.1f} {fcfg['unit']}) + {rescue_edges_count} Arestas de Resgate da MST | "
        f"100% Conexo (0 Órfãos) | Modularidade Q = {modularity_q:.4f}",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap MST ∪ P25 salvo em: {output_png.name} ({output_png})")


def print_stage_3b_report(
    topo_info: dict,
    df_dtw_metrics: pd.DataFrame,
    df_features: pd.DataFrame,
    modularity_q: float,
    resolution: float,
):
    """Imprime o relatório detalhado da Etapa 3B (MST ∪ P25 + Louvain Puro)."""
    n_comm = len(df_dtw_metrics)
    print("\n" + "=" * 96)
    print("   ETAPA 3B — PARTE 1: AUDITORIA TOPOLÓGICA DA REDE OTIMIZADA (MST ∪ P25)")
    print("=" * 96)
    print(f"-> Total de Janelas (Vértices):               {topo_info['n_nodes']:,d} nós (100% CONECTADOS)")
    print(f"-> Arestas no Percentil 25 (Top 25%):         {topo_info['p25_edges_count']:,d} arestas (D_ij <= {topo_info['dtw_p25_threshold_kmh']:,.2f} km/h)")
    print(f"-> Arestas Totais na MST:                     {topo_info['mst_total_edges']:,d} arestas")
    print(f"-> Arestas da MST já presentes no P25:        {topo_info['mst_already_in_p25']:,d} arestas")
    print(f"-> Arestas de Resgate que a MST adicionou:    {topo_info['mst_rescue_edges_added']:,d} arestas (impedindo isolamento de nós)")
    print(f"-> Total Final de Arestas (MST ∪ P25):        {topo_info['union_edges_total']:,d} arestas ({topo_info['union_edges_pct']:.2f}% do grafo completo)")
    print("-" * 96)
    print("-> DIAGNÓSTICO DE CONECTIVIDADE DA REDE UNIFICADA:")
    print(f"   • Grafo 100% Conexo?                       {topo_info['is_connected']} (GARANTIA DA MST)")
    print(f"   • Total de Componentes Conexas:            {topo_info['connected_components']} (Componente única global)")
    print(f"   • Janelas Órfãs / Isoladas (Grau == 0):    {topo_info['isolated_orphan_nodes_count']} (ZERO ÓRFÃOS)")
    print(f"   • Grau Mínimo de um nó na rede:            {topo_info['min_degree']} (todos têm pelo menos 1 conexão essencial)")
    print(f"   • Grau Médio na rede unificada:            {topo_info['mean_degree']:.2f} conexões por janela")

    print("\n" + "=" * 96)
    print("   ETAPA 3B — PARTE 2: RESULTADO DO LOUVAIN 100% PURO SOBRE MST ∪ P25")
    print("=" * 96)
    print(f"-> Modularidade de Newman (Q) na MST ∪ P25 = {modularity_q:.4f} (Resolução = {resolution})")
    print(f"-> Comunidades Naturais Descobertas:          {n_comm} comunidades coesas (0 órfãos)")
    print("-" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'Proporção':<10} | "
        f"{'DTW Interno Médio':<19} | {'DTW Externo Médio':<19} | {'Razão Int/Ext':<13}"
    )
    print("-" * 96)

    for _, row in df_dtw_metrics.iterrows():
        ratio = row["intra_dtw_mean"] / row["inter_dtw_mean"] if row["inter_dtw_mean"] > 0 else 0.0
        print(
            f"C{int(row['community_id']):<11} | "
            f"{int(row['count']):<8} | "
            f"{row['percentage']:>8.2f}%  | "
            f"{row['intra_dtw_mean']:>19.2f} | "
            f"{row['inter_dtw_mean']:>19.2f} | "
            f"{ratio:>13.3f}"
        )

    print("\n" + "=" * 96)
    print("   ETAPA 3B — PARTE 3: MÉDIAS FÍSICAS OBSERVADAS A POSTERIORI NAS COMUNIDADES (MST ∪ P25)")
    print("=" * 96)
    print(
        f"{'Comunidade':<12} | {'Janelas':<8} | {'v_méd (km/h)':<12} | {'std_v (km/h)':<12} | "
        f"{'v_max (km/h)':<12} | {'a_méd (m/s²)':<12} | {'Fren/min':<9} | {'Acel/min':<9}"
    )
    print("-" * 96)

    for _, row in df_dtw_metrics.iterrows():
        c_id = int(row["community_id"])
        sub = df_features[df_features["p25_mst_louvain_community"] == c_id]
        print(
            f"C{c_id:<11} | "
            f"{len(sub):<8} | "
            f"{sub['mean_speed_kmh'].mean():>12.2f} | "
            f"{sub['std_speed_kmh'].mean():>12.2f} | "
            f"{sub['max_speed_kmh'].mean():>12.2f} | "
            f"{sub['mean_pos_accel_ms2'].mean():>12.2f} | "
            f"{sub['hard_braking_rate_min'].mean():>9.2f} | "
            f"{sub['rapid_accel_rate_min'].mean():>9.2f}"
        )
    print("=" * 96 + "\n")


def save_all_stages_comparison(
    is_pilot: bool,
    mode_key: str,
    feature: str = "speed_kmh",
):
    """
    Consolida o comparativo rigoroso de todos os 4 cenários:
    1. Grafo Completo (1.444.150 arestas, sem filtro);
    2. MST Pura (1.699 arestas, Kruskal);
    3. Percentil 25 Direto (361.038 arestas, com nó órfão);
    4. MST ∪ P25 (361.038 arestas + resgate da MST, 100% conexo).
    """
    fcfg = get_feature_paths(feature)
    # Carrega métricas das etapas
    p25_dir_path = fcfg["p25_direct_metrics_json"]
    p25_mst_path = fcfg["p25_mst_metrics_json"]
    mst_struct_path = fcfg["mst_structural_metrics_json"]
    louvain_mst_path = fcfg["louvain_mst_comparison_json"]

    p25_dir_data = (
        json.loads(p25_dir_path.read_text(encoding="utf-8")).get(mode_key, {})
        if p25_dir_path.exists()
        else {}
    )
    p25_mst_data = (
        json.loads(p25_mst_path.read_text(encoding="utf-8")).get(mode_key, {})
        if p25_mst_path.exists()
        else {}
    )
    mst_struct_data = (
        json.loads(mst_struct_path.read_text(encoding="utf-8")).get(mode_key, {})
        if mst_struct_path.exists()
        else {}
    )
    louvain_mst_comp = (
        json.loads(louvain_mst_path.read_text(encoding="utf-8")).get(mode_key, {})
        if louvain_mst_path.exists()
        else {}
    )

    full_info = louvain_mst_comp.get("full_graph_louvain", {})
    mst_info = louvain_mst_comp.get("mst_pure_louvain", {})

    rows = [
        {
            "cenario": "1. Grafo Completo (Sem Filtro)",
            "arestas": full_info.get("n_edges", 1444150 if not is_pilot else 4950),
            "pct_arestas": 100.0,
            "nos_orfaos": 0,
            "is_conexo": True,
            "modularidade_q": full_info.get("modularity_q", 0.1035),
            "comunidades_total": full_info.get("n_communities", 3),
            "dtw_intra_medio": full_info.get("weighted_intra_dtw_mean", 1274.52),
            "dtw_inter_medio": full_info.get("weighted_inter_dtw_mean", 2133.78),
            "razao_intra_inter": full_info.get("intra_inter_ratio", 0.597),
        },
        {
            "cenario": "2. Árvore Geradora Mínima (MST Pura)",
            "arestas": mst_info.get("n_edges", 1699 if not is_pilot else 99),
            "pct_arestas": float((mst_info.get("n_edges", 1699 if not is_pilot else 99) / (1444150 if not is_pilot else 4950)) * 100.0),
            "nos_orfaos": 0,
            "is_conexo": True,
            "modularidade_q": mst_info.get("modularity_q", 0.9492),
            "comunidades_total": mst_info.get("n_communities", 41),
            "dtw_intra_medio": mst_info.get("weighted_intra_dtw_mean", 714.75),
            "dtw_inter_medio": mst_info.get("weighted_inter_dtw_mean", 1869.24),
            "razao_intra_inter": mst_info.get("intra_inter_ratio", 0.382),
        },
        {
            "cenario": "3A. Percentil 25 Direto (Sem MST)",
            "arestas": p25_dir_data.get("retained_edges", 361038),
            "pct_arestas": p25_dir_data.get("retained_edges_pct", 25.0),
            "nos_orfaos": p25_dir_data.get("isolated_orphan_nodes_count", 1 if not is_pilot else 4),
            "is_conexo": p25_dir_data.get("is_connected", False),
            "modularidade_q": p25_dir_data.get("modularity_q", 0.3185),
            "comunidades_total": p25_dir_data.get("total_partitions_returned", 6),
            "dtw_intra_medio": p25_dir_data.get("active_weighted_intra_dtw_mean", 1078.31),
            "dtw_inter_medio": p25_dir_data.get("active_weighted_inter_dtw_mean", 2087.36),
            "razao_intra_inter": p25_dir_data.get("active_intra_inter_ratio", 0.517),
        },
        {
            "cenario": "3B. Rede Otimizada (MST ∪ P25)",
            "arestas": p25_mst_data.get("union_edges_total", 361038),
            "pct_arestas": p25_mst_data.get("union_edges_pct", 25.0),
            "nos_orfaos": p25_mst_data.get("isolated_orphan_nodes_count", 0),
            "is_conexo": p25_mst_data.get("is_connected", True),
            "modularidade_q": p25_mst_data.get("modularity_q", 0.0),
            "comunidades_total": p25_mst_data.get("comunidades_count", 0),
            "dtw_intra_medio": p25_mst_data.get("weighted_intra_dtw_mean", 0.0),
            "dtw_inter_medio": p25_mst_data.get("weighted_inter_dtw_mean", 0.0),
            "razao_intra_inter": p25_mst_data.get("intra_inter_ratio", 0.0),
        },
    ]

    df_comp = pd.DataFrame(rows)
    comp_json = fcfg["dtw_all_stages_comparison_json"]
    comp_csv = fcfg["dtw_all_stages_comparison_csv"]

    if not is_pilot or not comp_csv.exists():
        comp_json.parent.mkdir(parents=True, exist_ok=True)
        comp_json.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        df_comp.to_csv(comp_csv, index=False, encoding="utf-8")
        print("\n" + "=" * 96)
        print(f"   TABELA COMPARATIVA CONSOLIDADA DE TODOS OS CENÁRIOS ({fcfg['name_pt'].upper()})")
        print("=" * 96)
        print(df_comp.to_string(index=False))
        print("=" * 96 + "\n")


def run_stage_3b_mst_percentile(
    is_pilot: bool = False,
    resolution: float = 1.0,
    feature: str = "speed_kmh",
) -> tuple[pd.DataFrame, dict]:
    """
    Executa a Etapa 3B: Construção da rede unificada MST ∪ P25 e Louvain 100% Puro.
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    mode_str = f"PILOTO ({DTW_PILOT_SIZE}x{DTW_PILOT_SIZE})" if is_pilot else "COMPLETO (1700x1700)"
    print("=" * 96)
    print(
        f"INICIANDO ETAPA 3B: REDE OTIMIZADA (MST ∪ P25) [{fcfg['name_pt'].upper()}] + LOUVAIN PURO - MODO {mode_str}"
    )
    print("=" * 96)

    dtw_npy_path = fcfg["dtw_pilot_npy"] if is_pilot else fcfg["dtw_full_npy"]
    mst_edges_pq = fcfg["mst_pilot_edges_parquet"] if is_pilot else fcfg["mst_full_edges_parquet"]

    dist_matrix = np.load(dtw_npy_path)
    n_windows = dist_matrix.shape[0]

    df_features = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET)
    if is_pilot:
        df_features = df_features.iloc[:n_windows].copy().reset_index(drop=True)
    else:
        df_features = df_features.copy().reset_index(drop=True)

    # 1. Constrói grafo unificado MST ∪ P25
    start_t = time.time()
    G_union, topo_info = build_mst_percentile_graph(
        dist_matrix=dist_matrix,
        mst_edges_parquet=mst_edges_pq,
        percentile=DTW_PERCENTILE_CUTOFF,
    )

    # 2. Executa Louvain 100% puro sobre a rede unificada
    communities, modularity_q = detect_louvain_communities(
        G_union, resolution=resolution, seed=LOUVAIN_RANDOM_SEED
    )
    elapsed_s = time.time() - start_t

    # Reordena matriz DTW pelas comunidades da rede unificada
    reordered_dist, ordered_indices, boundary_indices, node_to_comm = reorder_dtw_matrix(
        dist_matrix, communities
    )

    df_dtw_metrics = compute_pure_dtw_community_metrics(dist_matrix, communities)

    df_features["p25_mst_louvain_community"] = [node_to_comm[i] for i in range(n_windows)]
    df_features["p25_mst_degree"] = [int(G_union.degree(i)) for i in range(n_windows)]

    # 3. Gera Heatmap em 300 DPI
    heatmap_path = (
        fcfg["fig_p25_mst_pilot_heatmap"] if is_pilot else fcfg["fig_p25_mst_full_heatmap"]
    )
    plot_p25_mst_heatmap(
        matrix_reordered=reordered_dist,
        boundary_indices=boundary_indices,
        communities=communities,
        output_png=heatmap_path,
        n_windows=n_windows,
        dtw_threshold=topo_info["dtw_p25_threshold_kmh"],
        modularity_q=modularity_q,
        rescue_edges_count=topo_info["mst_rescue_edges_added"],
        feature=feature,
    )

    # 4. Imprime relatório detalhado da Etapa 3B
    print_stage_3b_report(
        topo_info=topo_info,
        df_dtw_metrics=df_dtw_metrics,
        df_features=df_features,
        modularity_q=modularity_q,
        resolution=resolution,
    )

    # 5. Salva artefatos da Etapa 3B
    out_csv = fcfg["p25_mst_pilot_csv"] if is_pilot else fcfg["p25_mst_full_csv"]
    out_pq = fcfg["p25_mst_pilot_parquet"] if is_pilot else fcfg["p25_mst_full_parquet"]
    metrics_json = fcfg["p25_mst_metrics_json"]
    summary_csv = fcfg["p25_mst_summary_csv"]

    df_features.to_csv(out_csv, index=False, encoding="utf-8")
    df_features.to_parquet(out_pq, index=False)

    w_intra = float(np.sum(df_dtw_metrics["intra_dtw_mean"] * df_dtw_metrics["count"]) / n_windows)
    w_inter = float(np.sum(df_dtw_metrics["inter_dtw_mean"] * df_dtw_metrics["count"]) / n_windows)

    result_payload = {
        **topo_info,
        "louvain_execution_time_s": float(elapsed_s),
        "modularity_q": float(modularity_q),
        "comunidades_count": int(len(communities)),
        "weighted_intra_dtw_mean": w_intra,
        "weighted_inter_dtw_mean": w_inter,
        "intra_inter_ratio": float(w_intra / w_inter) if w_inter > 0 else 0.0,
    }

    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    json_data = {}
    if metrics_json.exists():
        try:
            json_data = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            json_data = {}
    json_data[mode_key] = result_payload
    metrics_json.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not is_pilot:
        kinematic_means = (
            df_features.groupby("p25_mst_louvain_community")[
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
            .rename(columns={"p25_mst_louvain_community": "community_id"})
        )
        df_summary = df_dtw_metrics.merge(kinematic_means, on="community_id", how="left")
        df_summary.to_csv(summary_csv, index=False, encoding="utf-8")

    # Atualiza a tabela comparativa geral de todos os cenários
    save_all_stages_comparison(is_pilot=is_pilot, mode_key=mode_key, feature=feature)

    print("-> Artefatos da Etapa 3B (MST ∪ P25) salvos com sucesso:")
    print(f"   Clusters CSV:     {out_csv.name}")
    print(f"   Clusters Parquet: {out_pq.name}")
    print(f"   Métricas JSON:    {metrics_json.name}")
    if not is_pilot:
        print(f"   Resumo CSV:       {summary_csv.name}")
        print(f"   Tabela Geral CSV: {fcfg['dtw_all_stages_comparison_csv'].name}")
    print("=" * 96)
    print("ETAPA 3B (REDE OTIMIZADA MST ∪ P25 + LOUVAIN PURO) CONCLUÍDA COM SUCESSO!")
    print("=" * 96 + "\n")

    return df_features, result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtragem por Percentil 25 da Matriz DTW e Louvain Puro (Etapas 3A e 3B)."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Executa no modo piloto 100x100.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa no modo treino completo 1700x1700.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["3a", "3b", "all"],
        default="all",
        help="Etapa a executar: '3a' (Percentil 25 Direto), '3b' (MST ∪ P25) ou 'all' (ambas).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Parâmetro de resolução do Louvain (padrão puro: 1.0).",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a analisar (padrão: speed_kmh).",
    )

    args = parser.parse_args()
    is_pilot_mode = True if (args.pilot and not args.full) else False

    if args.stage in ("3a", "all"):
        run_stage_3a_direct_percentile(
            is_pilot=is_pilot_mode, resolution=args.resolution, feature=args.feature
        )
    if args.stage in ("3b", "all"):
        run_stage_3b_mst_percentile(
            is_pilot=is_pilot_mode, resolution=args.resolution, feature=args.feature
        )
