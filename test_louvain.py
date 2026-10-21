"""
Script de Teste e Validação da Detecção de Comunidades pelo Algoritmo de Louvain Puro.
Verifica:
1. Existência e integridade dos artefatos salvos (.csv, .parquet e heatmap .png);
2. Cobertura estrita de 100% dos nós/janelas (100 no piloto ou 1.700 na completa);
3. Ausência de rótulos heurísticos forçados e integridade da partição matemática;
4. Consistência e não-vacuidade das comunidades identificadas;
5. Validação da coesão interna na matriz DTW (distância DTW intra-comunidade < distância inter-comunidade).

Execute com: python test_louvain.py [--pilot | --full]

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import argparse

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.config import (
    DTW_PILOT_SIZE,
    DTW_PILOT_MATRIX_NPY,
    LOUVAIN_PILOT_CLUSTERS_CSV,
    LOUVAIN_PILOT_CLUSTERS_PARQUET,
    FIG_LOUVAIN_PILOT_HEATMAP,
    DTW_SPEED_MATRIX_NPY,
    LOUVAIN_FULL_CLUSTERS_CSV,
    LOUVAIN_FULL_CLUSTERS_PARQUET,
    FIG_LOUVAIN_FULL_HEATMAP,
    TRAIN_WINDOW_COUNT,
)


def test_louvain(mode: str = "pilot"):
    is_pilot = mode == "pilot"
    expected_count = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    dtw_npy_path = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    csv_path = LOUVAIN_PILOT_CLUSTERS_CSV if is_pilot else LOUVAIN_FULL_CLUSTERS_CSV
    parquet_path = LOUVAIN_PILOT_CLUSTERS_PARQUET if is_pilot else LOUVAIN_FULL_CLUSTERS_PARQUET
    fig_path = FIG_LOUVAIN_PILOT_HEATMAP if is_pilot else FIG_LOUVAIN_FULL_HEATMAP

    mode_label = f"PILOTO ({expected_count} nós)" if is_pilot else f"COMPLETA ({expected_count} nós)"
    print("=" * 72)
    print(f"INICIANDO TESTE DE VALIDAÇÃO DO LOUVAIN PURO - MODO {mode_label}")
    print("=" * 72)

    # 1. Existência dos artefatos
    print("\n[1/4] Verificando existência dos artefatos gerados pelo Louvain...")
    for p, name in [
        (csv_path, f"Tabela de Comunidades CSV ({csv_path.name})"),
        (parquet_path, f"Tabela de Comunidades Parquet ({parquet_path.name})"),
        (fig_path, f"Heatmap Reordenado PNG ({fig_path.name})"),
    ]:
        if not p.exists():
            print(f"[ERRO] Arquivo {name} não encontrado em: {p}")
            sys.exit(1)
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {name}: Encontrado! ({size_kb:.1f} KB)")

    # 2. Carregar datasets e validar contagens e pureza
    print(f"\n[2/4] Validando cobertura de 100% dos nós e ausência de rótulos forçados...")
    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(parquet_path)

    assert len(df_csv) == expected_count, (
        f"Contagem incorreta no CSV: esperado {expected_count}, obteve {len(df_csv)}"
    )
    assert len(df_pq) == expected_count, (
        f"Contagem incorreta no Parquet: esperado {expected_count}, obteve {len(df_pq)}"
    )

    assert "louvain_community" in df_csv.columns, "Coluna 'louvain_community' ausente no CSV!"
    assert "louvain_interpretation" not in df_csv.columns, (
        "Coluna heurística 'louvain_interpretation' não deve existir no modo puro!"
    )
    print(f"  [OK] {expected_count} nós validados e livres de rótulos heurísticos forçados.")

    # 3. Consistência matemática das comunidades e coesão DTW
    print("\n[3/4] Validando coesão interna na matriz DTW (DTW Intra < DTW Inter)...")
    dist_matrix = np.load(dtw_npy_path)
    communities = sorted(df_csv["louvain_community"].unique())
    n_communities = len(communities)

    assert n_communities >= 2, f"Número insuficiente de comunidades detectadas: {n_communities}"
    assert communities == list(range(n_communities)), (
        f"Comunidades devem ser indexadas de 0 a {n_communities - 1}, obtido: {communities}"
    )

    all_nodes = set(range(expected_count))
    for c_id in communities:
        nodes = df_csv.index[df_csv["louvain_community"] == c_id].tolist()
        other_nodes = sorted(list(all_nodes - set(nodes)))
        assert len(nodes) > 0, f"Comunidade C{c_id} vazia!"

        sub_intra = dist_matrix[np.ix_(nodes, nodes)]
        iu = np.triu_indices(len(nodes), k=1)
        intra_mean = float(sub_intra[iu].mean())
        inter_mean = float(dist_matrix[np.ix_(nodes, other_nodes)].mean())

        assert intra_mean < inter_mean, (
            f"Comunidade C{c_id} inválida: DTW interno ({intra_mean:.1f}) >= DTW externo ({inter_mean:.1f})!"
        )
        print(
            f"  [OK] Comunidade C{c_id} (N={len(nodes)}): "
            f"DTW Interno = {intra_mean:.1f} < DTW Externo = {inter_mean:.1f}"
        )

    # 4. Sumário estatístico observado a posteriori
    print("\n[4/4] Médias cinemáticas observadas a posteriori em cada comunidade:")
    summary = df_csv.groupby("louvain_community").agg(
        qtd=("window_file", "count"),
        v_med=("mean_speed_kmh", "mean"),
        std_v=("std_speed_kmh", "mean"),
        v_max=("max_speed_kmh", "mean"),
        fren_min=("hard_braking_rate_min", "mean"),
        acel_min=("rapid_accel_rate_min", "mean"),
    )
    for c_id, row in summary.iterrows():
        pct = (row["qtd"] / expected_count) * 100.0
        print(
            f"  • C{c_id}: {int(row['qtd'])} janelas ({pct:.1f}%) | "
            f"v_méd = {row['v_med']:.2f} km/h | std_v = {row['std_v']:.2f} km/h | "
            f"fren/min = {row['fren_min']:.2f} | acel/min = {row['acel_min']:.2f}"
        )

    print("\n" + "=" * 72)
    print("TODOS OS TESTES DO LOUVAIN PURO PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa e valida os resultados do Algoritmo de Louvain Puro.")
    parser.add_argument("--pilot", action="store_true", default=True, help="Valida o modo piloto (100 nós).")
    parser.add_argument("--full", action="store_true", help="Valida o modo completo (1700 nós).")

    args = parser.parse_args()
    mode = "full" if args.full else "pilot"
    test_louvain(mode=mode)
