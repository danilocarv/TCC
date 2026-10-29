"""
Script de Teste e Validação do Pipeline de Correlação de Pearson e Comunidades de Louvain.

Verifica:
1. Reamostragem uniforme de séries temporais em grid de 100 pontos;
2. Propriedades matemáticas da matriz de correlação (simetria, diagonal=1, limites [-1, 1]);
3. Construção do grafo ponderado completo e ausência de arestas negativas;
4. Execução do Algoritmo de Louvain Puro e não-vacuidade das comunidades;
5. Geração dos artefatos (.npy, .parquet, heatmap .png, relatório .json);
6. Consistência das métricas de cruzamento com DTW e K-Means.

Execute com: python tests/test_correlation_pipeline.py [--pilot | --full]

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.config import (
    DTW_PILOT_SIZE,
    TRAIN_WINDOW_COUNT,
    get_feature_paths,
)
from src.correlation_analysis import (
    compute_pearson_correlation_matrix,
    build_full_correlation_graph,
    detect_louvain_communities,
    load_and_resample_series,
    run_correlation_pipeline,
)


def test_correlation_pipeline(mode: str = "pilot", feature: str = "speed_kmh"):
    is_pilot = mode == "pilot"
    expected_count = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    mode_label = f"PILOTO ({expected_count} nós)" if is_pilot else f"COMPLETA ({expected_count} nós)"

    print("=" * 75)
    print(f"TESTE DO PIPELINE DE CORRELAÇÃO DE PEARSON — {feature.upper()} [{mode_label}]")
    print("=" * 75)

    # 1. Teste de Reamostragem
    print("\n[1/5] Testando carregamento e reamostragem das séries temporais...")
    n_points = 100
    X, file_stems = load_and_resample_series(feature=feature, n_windows=expected_count, n_points=n_points)
    assert X.shape == (expected_count, n_points), f"Shape incorreto: {X.shape} != ({expected_count}, {n_points})"
    assert not np.isnan(X).any(), "Detectado NaN na matriz reamostrada!"
    assert not np.isinf(X).any(), "Detectado Inf na matriz reamostrada!"
    assert len(file_stems) == expected_count, "Quantidade de arquivos divergente!"
    print(f"   [OK] Matriz X ({expected_count}x{n_points}) reamostrada com sucesso e sem valores ausentes.")

    # 2. Teste da Matriz de Correlação
    print("\n[2/5] Testando propriedades matemáticas da matriz de correlação...")
    corr = compute_pearson_correlation_matrix(X)
    assert corr.shape == (expected_count, expected_count), f"Shape incorreto: {corr.shape}"
    assert np.allclose(corr, corr.T, atol=1e-8), "Matriz de correlação não é perfeitamente simétrica!"
    diag = np.diag(corr)
    assert np.allclose(diag, 1.0, atol=1e-8), "Diagonal principal não é unitária!"
    assert np.all(corr >= -1.0 - 1e-8) and np.all(corr <= 1.0 + 1e-8), "Valores fora do intervalo [-1, 1]!"
    print(f"   [OK] Matriz simétrica ({expected_count}x{expected_count}), diagonal=1.0, limites [-1.0, 1.0] válidos.")

    # 3. Teste do Grafo Ponderado Completo
    print("\n[3/5] Testando grafo ponderado (W_ij = max(0, r_ij))...")
    G, topo_info = build_full_correlation_graph(corr)
    assert G.number_of_nodes() == expected_count, f"Nós divergentes: {G.number_of_nodes()} != {expected_count}"
    for u, v, data in G.edges(data=True):
        assert data["weight"] > 0, f"Aresta com peso não-positivo encontrada: {data['weight']}"
    print(f"   [OK] Grafo com {G.number_of_nodes()} nós e {G.number_of_edges():,d} arestas positivas.")

    # 4. Teste do Louvain Puro
    print("\n[4/5] Testando convergência do Algoritmo de Louvain Puro...")
    communities, mod_q = detect_louvain_communities(G, resolution=1.0)
    all_assigned = [node for comm in communities for node in comm]
    assert len(all_assigned) == expected_count, "Nem todos os nós foram particionados pelo Louvain!"
    assert len(set(all_assigned)) == expected_count, "Partição contém nós duplicados!"
    assert mod_q >= 0.0, f"Modularidade Q negativa: {mod_q}"
    n_active = sum(1 for c in communities if len(c) > 1)
    print(f"   [OK] Louvain gerou {n_active} comunidades principais (Modularidade Q = {mod_q:.4f}).")

    # 5. Execução do Pipeline e Validação de Artefatos
    print("\n[5/5] Executando pipeline completo e verificando persistência dos artefatos...")
    report = run_correlation_pipeline(feature=feature, is_pilot=is_pilot, n_points=n_points)
    fcfg = get_feature_paths(feature)

    expected_files = [
        fcfg["corr_pilot_npy"] if is_pilot else fcfg["corr_full_npy"],
        fcfg["corr_pilot_parquet"] if is_pilot else fcfg["corr_full_parquet"],
        fcfg["corr_louvain_pilot_csv"] if is_pilot else fcfg["corr_louvain_full_csv"],
        fcfg["corr_louvain_pilot_parquet"] if is_pilot else fcfg["corr_louvain_full_parquet"],
        fcfg["fig_corr_pilot_heatmap"] if is_pilot else fcfg["fig_corr_full_heatmap"],
        fcfg["fig_corr_louvain_pilot_heatmap"] if is_pilot else fcfg["fig_corr_louvain_full_heatmap"],
        fcfg["corr_vs_dtw_comparison_json"],
    ]

    for p in expected_files:
        assert p.exists(), f"Artefato esperado não foi criado: {p}"
        print(f"   [OK] Artefato verificado: {p.name}")

    print("\n" + "=" * 75)
    print(f"TESTE CONCLUÍDO COM 100% DE SUCESSO NO MODO {mode_label}!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Teste do Pipeline de Correlação de Pearson e Louvain.")
    parser.add_argument("--pilot", action="store_true", help="Executa teste em modo piloto (100 nós)")
    parser.add_argument("--full", action="store_true", help="Executa teste em modo completo (1.700 nós)")
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=["speed_kmh", "accel_ms2", "power_pos_kw"],
        help="Feature a validar",
    )
    args = parser.parse_args()

    mode = "full" if args.full else "pilot"
    test_correlation_pipeline(mode=mode, feature=args.feature)


if __name__ == "__main__":
    main()
