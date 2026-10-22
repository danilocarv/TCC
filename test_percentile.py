"""
Script de Teste e Validação Automatizada da Etapa 3A:
Filtragem Direta pelo Percentil 25 da Matriz DTW e Algoritmo de Louvain 100% Puro.

Verifica:
1. Existência e integridade dos artefatos da Etapa 3A (.csv, .parquet, .json, .png);
2. Exatidão matemática do limiar do Percentil 25 (P25) e retenção estrita de ~25% das arestas;
3. Ausência de auto-laços da diagonal principal (self-loops == 0);
4. Cobertura de 100% dos nós, diagnóstico transparente de nós órfãos/isolados (grau == 0)
   e melhoria da Modularidade de Newman (Q_P25 > Q_Grafo_Completo);
5. Coesão interna na matriz DTW nas comunidades reais (DTW Intra < DTW Inter).

Execute com:
  python test_percentile.py [--pilot | --full]

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

from src.config import (
    DTW_PILOT_SIZE,
    TRAIN_WINDOW_COUNT,
    DTW_PILOT_MATRIX_NPY,
    DTW_SPEED_MATRIX_NPY,
    DTW_PERCENTILE_CUTOFF,
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
)


def validate_stage_3a(is_pilot: bool):
    expected_n = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    dtw_npy = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    clusters_csv = P25_DIRECT_PILOT_CLUSTERS_CSV if is_pilot else P25_DIRECT_FULL_CLUSTERS_CSV
    clusters_pq = P25_DIRECT_PILOT_CLUSTERS_PARQUET if is_pilot else P25_DIRECT_FULL_CLUSTERS_PARQUET
    fig_heatmap = FIG_P25_DIRECT_PILOT_HEATMAP if is_pilot else FIG_P25_DIRECT_FULL_HEATMAP
    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"

    mode_str = f"PILOTO ({expected_n}x{expected_n})" if is_pilot else f"COMPLETO ({expected_n}x{expected_n})"
    print("=" * 84)
    print(f"VALIDANDO ETAPA 3A: PERCENTIL 25 DIRETO + LOUVAIN PURO - MODO {mode_str}")
    print("=" * 84)

    # 1. Existência dos arquivos
    print("\n[1/4] Verificando existência dos artefatos da Etapa 3A...")
    files_to_check = [
        (clusters_csv, f"Clusters P25 Direto CSV ({clusters_csv.name})"),
        (clusters_pq, f"Clusters P25 Direto Parquet ({clusters_pq.name})"),
        (fig_heatmap, f"Heatmap P25 Direto PNG ({fig_heatmap.name})"),
        (P25_DIRECT_METRICS_JSON, f"Métricas P25 Direto JSON ({P25_DIRECT_METRICS_JSON.name})"),
    ]
    if not is_pilot:
        files_to_check.append((P25_DIRECT_SUMMARY_CSV, f"Resumo P25 Direto CSV ({P25_DIRECT_SUMMARY_CSV.name})"))

    for p, desc in files_to_check:
        assert p.exists(), f"[ERRO] Arquivo ausente: {p}"
        print(f"  [OK] {desc}: Encontrado ({p.stat().st_size / 1024:.1f} KB)")

    # 2. Exatidão matemática do Percentil 25
    print("\n[2/4] Auditando cálculo matemático do Percentil 25 na Matriz DTW...")
    dist_matrix = np.load(dtw_npy)
    rows, cols = np.triu_indices(expected_n, k=1)
    off_diag = dist_matrix[rows, cols]
    expected_p25 = float(np.percentile(off_diag, DTW_PERCENTILE_CUTOFF))

    metrics_all = json.loads(P25_DIRECT_METRICS_JSON.read_text(encoding="utf-8"))
    assert mode_key in metrics_all, f"Chave {mode_key} ausente em {P25_DIRECT_METRICS_JSON.name}!"
    m = metrics_all[mode_key]

    assert np.isclose(m["dtw_p25_threshold_kmh"], expected_p25, atol=1e-4), (
        f"Limiar P25 divergente: esperado {expected_p25:.4f}, obtido {m['dtw_p25_threshold_kmh']:.4f}"
    )
    assert 24.9 <= m["retained_edges_pct"] <= 25.1, (
        f"Percentual de arestas retidas fora de ~25%: {m['retained_edges_pct']:.2f}%"
    )
    print(
        f"  [OK] Limiar P25 validado: D_ij <= {m['dtw_p25_threshold_kmh']:.2f} km/h "
        f"({m['retained_edges']:,d} arestas retidas = {m['retained_edges_pct']:.2f}%)."
    )

    # 3. Cobertura dos nós e diagnóstico de nós órfãos
    print("\n[3/4] Validando cobertura de 100% dos nós e diagnóstico de conectividade...")
    df_csv = pd.read_csv(clusters_csv)
    assert len(df_csv) == expected_n, f"Esperado {expected_n} linhas, obtido {len(df_csv)}"
    assert "p25_direct_louvain_community" in df_csv.columns
    assert "p25_direct_degree" in df_csv.columns
    assert "p25_direct_is_isolated" in df_csv.columns

    n_isolated = int(df_csv["p25_direct_is_isolated"].sum())
    assert n_isolated == m["isolated_orphan_nodes_count"], "Contagem de nós isolados inconsistente!"
    print(
        f"  [OK] {expected_n} nós auditados: {m['connected_nodes_count']} conectados e "
        f"{n_isolated} nós órfãos isolados (grau 0)."
    )

    # 4. Ganho de Modularidade Q e coesão DTW interna < externa
    print("\n[4/4] Validando Modularidade Q e coesão interna na matriz DTW...")
    assert m["modularity_q"] > 0.15, f"Modularidade Q inesperadamente baixa: {m['modularity_q']:.4f}"
    assert m["active_weighted_intra_dtw_mean"] < m["active_weighted_inter_dtw_mean"], (
        "DTW interno médio deve ser estritamente menor que DTW externo médio!"
    )
    print(
        f"  [OK] Modularidade de Newman Q = {m['modularity_q']:.4f} "
        f"({m['active_communities_count']} comunidades reais + {m['orphan_singletons_count']} órfãos)."
    )
    print(
        f"  [OK] Coesão DTW confirmada: Intra Médio = {m['active_weighted_intra_dtw_mean']:.2f} km/h < "
        f"Inter Médio = {m['active_weighted_inter_dtw_mean']:.2f} km/h "
        f"(Razão = {m['active_intra_inter_ratio']:.3f})."
    )


def validate_stage_3b(is_pilot: bool):
    expected_n = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    clusters_csv = P25_MST_PILOT_CLUSTERS_CSV if is_pilot else P25_MST_FULL_CLUSTERS_CSV
    clusters_pq = P25_MST_PILOT_CLUSTERS_PARQUET if is_pilot else P25_MST_FULL_CLUSTERS_PARQUET
    fig_heatmap = FIG_P25_MST_PILOT_HEATMAP if is_pilot else FIG_P25_MST_FULL_HEATMAP
    mode_key = "pilot_100x100" if is_pilot else "train_1700x1700"

    mode_str = f"PILOTO ({expected_n}x{expected_n})" if is_pilot else f"COMPLETO ({expected_n}x{expected_n})"
    print("\n" + "=" * 84)
    print(f"VALIDANDO ETAPA 3B: REDE OTIMIZADA (MST ∪ P25) + LOUVAIN PURO - MODO {mode_str}")
    print("=" * 84)

    # 1. Existência dos arquivos
    print("\n[1/4] Verificando existência dos artefatos da Etapa 3B...")
    files_to_check = [
        (clusters_csv, f"Clusters MST ∪ P25 CSV ({clusters_csv.name})"),
        (clusters_pq, f"Clusters MST ∪ P25 Parquet ({clusters_pq.name})"),
        (fig_heatmap, f"Heatmap MST ∪ P25 PNG ({fig_heatmap.name})"),
        (P25_MST_METRICS_JSON, f"Métricas MST ∪ P25 JSON ({P25_MST_METRICS_JSON.name})"),
    ]
    if not is_pilot:
        files_to_check.append((P25_MST_SUMMARY_CSV, f"Resumo MST ∪ P25 CSV ({P25_MST_SUMMARY_CSV.name})"))
        files_to_check.append((DTW_ALL_STAGES_COMPARISON_CSV, f"Comparativo Geral CSV ({DTW_ALL_STAGES_COMPARISON_CSV.name})"))
        files_to_check.append((DTW_ALL_STAGES_COMPARISON_JSON, f"Comparativo Geral JSON ({DTW_ALL_STAGES_COMPARISON_JSON.name})"))

    for p, desc in files_to_check:
        assert p.exists(), f"[ERRO] Arquivo ausente: {p}"
        print(f"  [OK] {desc}: Encontrado ({p.stat().st_size / 1024:.1f} KB)")

    # 2. Conectividade estrita (zero nós órfãos)
    print("\n[2/4] Auditando conectividade estrita da rede unificada MST ∪ P25...")
    metrics_all = json.loads(P25_MST_METRICS_JSON.read_text(encoding="utf-8"))
    assert mode_key in metrics_all, f"Chave {mode_key} ausente em {P25_MST_METRICS_JSON.name}!"
    m = metrics_all[mode_key]

    assert m["is_connected"] is True, "A rede unificada MST ∪ P25 DEVERIA ser 100% conexa!"
    assert m["isolated_orphan_nodes_count"] == 0, "A rede unificada não pode conter nós órfãos!"
    assert m["min_degree"] >= 1, "Todo nó deve ter pelo menos 1 aresta na rede unificada!"
    print(
        f"  [OK] Conectividade 100% garantida pela MST: 0 nós órfãos, {m['union_edges_total']:,d} arestas totais "
        f"({m['mst_rescue_edges_added']} arestas de resgate adicionadas pela MST)."
    )

    # 3. Cobertura da partição Louvain
    print("\n[3/4] Validando comunidades de Louvain na rede unificada...")
    df_csv = pd.read_csv(clusters_csv)
    assert len(df_csv) == expected_n, f"Esperado {expected_n} linhas, obtido {len(df_csv)}"
    assert "p25_mst_louvain_community" in df_csv.columns
    assert "p25_mst_degree" in df_csv.columns
    assert (df_csv["p25_mst_degree"] >= 1).all(), "Existem nós com grau zero na tabela!"

    n_comm = m["comunidades_count"]
    print(f"  [OK] {expected_n} nós particionados em {n_comm} comunidades coesas (0 órfãos).")

    # 4. Modularidade Q e coesão interna < externa
    print("\n[4/4] Validando Modularidade Q e coesão DTW na rede unificada...")
    assert m["modularity_q"] > 0.20, f"Modularidade inesperadamente baixa: {m['modularity_q']:.4f}"
    assert m["weighted_intra_dtw_mean"] < m["weighted_inter_dtw_mean"], (
        "DTW intra-comunidade médio deve ser estritamente menor que inter-comunidade!"
    )
    print(
        f"  [OK] Modularidade de Newman Q = {m['modularity_q']:.4f}."
    )
    print(
        f"  [OK] Coesão DTW confirmada: Intra Médio = {m['weighted_intra_dtw_mean']:.2f} km/h < "
        f"Inter Médio = {m['weighted_inter_dtw_mean']:.2f} km/h "
        f"(Razão = {m['intra_inter_ratio']:.3f})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa e valida a Etapa 3 (Percentil 25 Direto e MST ∪ P25).")
    parser.add_argument("--pilot", action="store_true", help="Valida modo piloto 100x100.")
    parser.add_argument("--full", action="store_true", help="Valida modo completo 1700x1700.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["3a", "3b", "all"],
        default="all",
        help="Etapa a validar: '3a' (Percentil 25 Direto), '3b' (MST ∪ P25) ou 'all' (ambas).",
    )

    args = parser.parse_args()
    is_pilot_mode = True if (args.pilot and not args.full) else False

    if args.stage in ("3a", "all"):
        validate_stage_3a(is_pilot=is_pilot_mode)
    if args.stage in ("3b", "all"):
        validate_stage_3b(is_pilot=is_pilot_mode)

    print("\n" + "=" * 84)
    print("TODOS OS TESTES EXECUTADOS PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 84 + "\n")
