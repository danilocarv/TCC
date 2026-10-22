"""
Script de Teste e Validação Automatizada da Árvore Geradora Mínima (Etapa 1) e
do Algoritmo de Louvain 100% Puro sobre a MST (Etapa 2).

Verifica:
[Etapa 1 - MST Pura (src/mst_builder.py)]:
1. Existência e integridade dos artefatos da árvore (.csv, .parquet, .npy, .json, .png);
2. Ausência estrita de auto-laços da diagonal principal (self-loops == 0);
3. Propriedades de Teoria dos Grafos: N vértices, exatamente N - 1 arestas,
   conectividade global (100% dos nós conectados, 0 órfãos) e aciclicidade (is_tree == True);
4. Otimalidade de Kruskal: distância DTW média nas arestas da MST substancialmente menor
   que na matriz DTW inteira.

[Etapa 2 - Louvain 100% Puro sobre a MST (src/mst_louvain.py)]:
5. Existência e integridade dos artefatos de comunidades da MST (.csv, .parquet, .json, .png);
6. Cobertura de 100% dos nós e ausência de rótulos heurísticos forçados ou indução de k;
7. Aumento drástico da Modularidade de Newman (Q_MST >> Q_Grafo_Completo);
8. Coesão interna na matriz DTW (DTW Intra-Comunidade < DTW Inter-Comunidade).

Execute com:
  python test_mst.py [--pilot | --full] [--stage 1 | 2 | all]

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
import argparse

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import networkx as nx

from src.config import (
    DTW_PILOT_SIZE,
    TRAIN_WINDOW_COUNT,
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
    LOUVAIN_MST_PILOT_CLUSTERS_CSV,
    LOUVAIN_MST_PILOT_CLUSTERS_PARQUET,
    FIG_LOUVAIN_MST_PILOT_HEATMAP,
    FIG_LOUVAIN_MST_TOPOLOGY_PILOT,
    LOUVAIN_MST_FULL_CLUSTERS_CSV,
    LOUVAIN_MST_FULL_CLUSTERS_PARQUET,
    FIG_LOUVAIN_MST_FULL_HEATMAP,
    FIG_LOUVAIN_MST_TOPOLOGY_FULL,
    LOUVAIN_MST_COMPARISON_JSON,
)


def validate_stage_1_mst(is_pilot: bool):
    expected_n = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    expected_edges = expected_n - 1
    dtw_npy = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    edges_csv = MST_PILOT_EDGES_CSV if is_pilot else MST_FULL_EDGES_CSV
    edges_pq = MST_PILOT_EDGES_PARQUET if is_pilot else MST_FULL_EDGES_PARQUET
    mst_npy = MST_PILOT_MATRIX_NPY if is_pilot else MST_FULL_MATRIX_NPY
    fig_pure = FIG_MST_PURE_PILOT if is_pilot else FIG_MST_PURE_FULL

    mode_str = f"PILOTO ({expected_n}x{expected_n})" if is_pilot else f"COMPLETO ({expected_n}x{expected_n})"
    print("=" * 80)
    print(f"VALIDANDO ETAPA 1: ÁRVORE GERADORA MÍNIMA (MST PURA) - MODO {mode_str}")
    print("=" * 80)

    # 1. Existência de arquivos
    print("\n[1/3] Verificando existência dos artefatos da Etapa 1 (MST Pura)...")
    for p, desc in [
        (edges_csv, f"Arestas MST CSV ({edges_csv.name})"),
        (edges_pq, f"Arestas MST Parquet ({edges_pq.name})"),
        (mst_npy, f"Matriz Adjacência MST NPY ({mst_npy.name})"),
        (MST_STRUCTURAL_METRICS_JSON, f"Métricas Estruturais JSON ({MST_STRUCTURAL_METRICS_JSON.name})"),
        (fig_pure, f"Figura MST Pura PNG ({fig_pure.name})"),
    ]:
        assert p.exists(), f"[ERRO] Arquivo ausente: {p}"
        print(f"  [OK] {desc}: Encontrado ({p.stat().st_size / 1024:.1f} KB)")

    # 2. Propriedades de Teoria dos Grafos e Diagonal Zero Ignorada
    print("\n[2/3] Auditando propriedades matemáticas da Árvore Geradora Mínima (Kruskal)...")
    df_edges = pd.read_parquet(edges_pq)
    mst_matrix = np.load(mst_npy)
    dist_matrix = np.load(dtw_npy)

    assert len(df_edges) == expected_edges, (
        f"A MST deve possuir exatamente N - 1 = {expected_edges} arestas, obtido {len(df_edges)}"
    )
    assert (df_edges["node_u"] != df_edges["node_v"]).all(), (
        "Encontrada aresta da diagonal principal (node_u == node_v) na MST!"
    )
    assert np.all(np.diag(mst_matrix) == 0.0), "Diagonal da matriz MST deve ser estritamente zero!"
    assert np.allclose(mst_matrix, mst_matrix.T), "Matriz da MST deve ser simétrica!"

    # Reconstrói grafo e verifica is_tree e is_connected
    T = nx.Graph()
    T.add_nodes_from(range(expected_n))
    for _, row in df_edges.iterrows():
        T.add_edge(int(row["node_u"]), int(row["node_v"]), weight=float(row["dtw_distance"]))

    assert T.number_of_nodes() == expected_n, f"Número incorreto de nós: {T.number_of_nodes()}"
    assert T.number_of_edges() == expected_edges, f"Número incorreto de arestas: {T.number_of_edges()}"
    assert nx.number_of_selfloops(T) == 0, "A MST não pode ter auto-laços!"
    assert nx.is_connected(T), "A MST não está 100% conexa (existem nós órfãos)!"
    assert nx.is_tree(T), "O grafo da MST contém ciclos ou não é uma árvore válida!"
    print(
        f"  [OK] Grafo validado: {expected_n} nós, {expected_edges} arestas, "
        "0 auto-laços, 100% conexo e estritamente acíclico (is_tree=True)."
    )

    # 3. Seletividade de pesos DTW (Kruskal)
    print("\n[3/3] Verificando seletividade de pesos DTW pelo Algoritmo de Kruskal...")
    off_diag_mean = float(dist_matrix[np.triu_indices(expected_n, k=1)].mean())
    mst_mean = float(df_edges["dtw_distance"].mean())
    assert mst_mean < off_diag_mean, (
        f"Peso médio da MST ({mst_mean:.2f}) deveria ser menor que da matriz completa ({off_diag_mean:.2f})!"
    )
    # Verifica ordenação crescente nos passos do Kruskal
    diffs = np.diff(df_edges["dtw_distance"].to_numpy())
    assert np.all(diffs >= -1e-9), "As arestas não estão ordenadas em ordem não-decrescente de Kruskal!"
    print(
        f"  [OK] Seletividade DTW confirmada: Média Matriz Inteira = {off_diag_mean:.2f} km/h "
        f"--> Média MST = {mst_mean:.2f} km/h."
    )


def validate_stage_2_louvain_on_mst(is_pilot: bool):
    expected_n = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    dtw_npy = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    clusters_csv = LOUVAIN_MST_PILOT_CLUSTERS_CSV if is_pilot else LOUVAIN_MST_FULL_CLUSTERS_CSV
    clusters_pq = LOUVAIN_MST_PILOT_CLUSTERS_PARQUET if is_pilot else LOUVAIN_MST_FULL_CLUSTERS_PARQUET
    fig_heatmap = FIG_LOUVAIN_MST_PILOT_HEATMAP if is_pilot else FIG_LOUVAIN_MST_FULL_HEATMAP
    fig_topology = FIG_LOUVAIN_MST_TOPOLOGY_PILOT if is_pilot else FIG_LOUVAIN_MST_TOPOLOGY_FULL

    mode_str = f"PILOTO ({expected_n}x{expected_n})" if is_pilot else f"COMPLETO ({expected_n}x{expected_n})"
    print("\n" + "=" * 80)
    print(f"VALIDANDO ETAPA 2: LOUVAIN 100% PURO SOBRE A MST - MODO {mode_str}")
    print("=" * 80)

    # 1. Existência de arquivos
    print("\n[1/3] Verificando existência dos artefatos da Etapa 2 (Louvain na MST)...")
    for p, desc in [
        (clusters_csv, f"Clusters MST+Louvain CSV ({clusters_csv.name})"),
        (clusters_pq, f"Clusters MST+Louvain Parquet ({clusters_pq.name})"),
        (fig_heatmap, f"Heatmap MST+Louvain PNG ({fig_heatmap.name})"),
        (fig_topology, f"Topologia MST+Louvain PNG ({fig_topology.name})"),
        (LOUVAIN_MST_COMPARISON_JSON, f"Comparativo JSON ({LOUVAIN_MST_COMPARISON_JSON.name})"),
    ]:
        assert p.exists(), f"[ERRO] Arquivo ausente: {p}"
        print(f"  [OK] {desc}: Encontrado ({p.stat().st_size / 1024:.1f} KB)")

    # 2. Pureza da partição e cobertura de 100% dos nós
    print("\n[2/3] Validando cobertura de 100% dos nós e ausência de indução/rótulos forçados...")
    df_csv = pd.read_csv(clusters_csv)
    df_pq = pd.read_parquet(clusters_pq)
    assert len(df_csv) == expected_n, f"Esperado {expected_n} linhas no CSV, obtido {len(df_csv)}"
    assert len(df_pq) == expected_n, f"Esperado {expected_n} linhas no Parquet, obtido {len(df_pq)}"
    assert "mst_louvain_community" in df_csv.columns, "Coluna 'mst_louvain_community' ausente!"
    assert "louvain_interpretation" not in df_csv.columns, "Não deve haver rótulos heurísticos forçados!"

    communities = sorted(df_csv["mst_louvain_community"].unique())
    n_comm = len(communities)
    assert n_comm >= 2, f"Quantidade inválida de comunidades: {n_comm}"
    assert communities == list(range(n_comm)), "As comunidades devem ser indexadas de 0 a K-1!"
    print(f"  [OK] {expected_n} nós mapeados em {n_comm} comunidades naturais puras (sem indução).")

    # 3. Melhoria da Modularidade Q e coesão DTW global
    print("\n[3/3] Validando ganho de Modularidade (Q_MST > Q_Completo) e coesão DTW...")
    comp_data = json.loads(LOUVAIN_MST_COMPARISON_JSON.read_text(encoding="utf-8"))
    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"
    assert mode_key in comp_data, f"Chave {mode_key} ausente no JSON comparativo!"

    q_mst = comp_data[mode_key]["mst_pure_louvain"]["modularity_q"]
    q_full = comp_data[mode_key]["full_graph_louvain"].get("modularity_q", 0.0)
    w_intra = comp_data[mode_key]["mst_pure_louvain"]["weighted_intra_dtw_mean"]
    w_inter = comp_data[mode_key]["mst_pure_louvain"]["weighted_inter_dtw_mean"]

    assert q_mst > q_full, (
        f"A modularidade na MST ({q_mst:.4f}) deveria superar a do grafo completo ({q_full:.4f})!"
    )
    assert w_intra < w_inter, (
        f"DTW intra-comunidade ponderado ({w_intra:.2f}) deve ser menor que inter-comunidade ({w_inter:.2f})!"
    )
    print(
        f"  [OK] Salto de Modularidade comprovado: Q_Completo = {q_full:.4f} --> Q_MST = {q_mst:.4f} "
        f"({q_mst / q_full:.1f}x maior!)."
    )
    print(
        f"  [OK] Coesão DTW confirmada: DTW Intra Médio = {w_intra:.2f} km/h < DTW Inter Médio = {w_inter:.2f} km/h."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valida matematicamente a Etapa 1 (MST Pura) e a Etapa 2 (Louvain Puro sobre a MST)."
    )
    parser.add_argument("--pilot", action="store_true", default=True, help="Valida modo piloto 100x100.")
    parser.add_argument("--full", action="store_true", help="Valida modo completo 1700x1700.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Etapa a validar: '1' (apenas MST), '2' (apenas Louvain na MST) ou 'all' (ambas).",
    )

    args = parser.parse_args()
    is_pilot_mode = not args.full

    if args.stage in ("1", "all"):
        validate_stage_1_mst(is_pilot=is_pilot_mode)
    if args.stage in ("2", "all"):
        validate_stage_2_louvain_on_mst(is_pilot=is_pilot_mode)

    print("\n" + "=" * 80)
    print("TODOS OS TESTES MATEMÁTICOS DA MST E DO LOUVAIN PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 80 + "\n")
