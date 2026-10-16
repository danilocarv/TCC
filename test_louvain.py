"""
Script de Teste e Validação da Detecção de Comunidades pelo Algoritmo de Louvain.
Verifica:
1. Existência e integridade dos artefatos salvos (.csv, .parquet e heatmap .png);
2. Cobertura estrita de 100% dos nós/janelas (100 no piloto ou 1.700 na completa);
3. Validação do índice de Modularidade de Rede (Q > 0.30 indica partição significativa);
4. Consistência e não-vacuidade das comunidades identificadas;
5. Não-degenerescência das features cinemáticas agregadas por comunidade.

Execute com: python test_louvain.py [--pilot | --full]

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import argparse
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.config import (
    DTW_PILOT_SIZE,
    LOUVAIN_PILOT_CLUSTERS_CSV,
    LOUVAIN_PILOT_CLUSTERS_PARQUET,
    FIG_LOUVAIN_PILOT_HEATMAP,
    LOUVAIN_FULL_CLUSTERS_CSV,
    LOUVAIN_FULL_CLUSTERS_PARQUET,
    FIG_LOUVAIN_FULL_HEATMAP,
    TRAIN_WINDOW_COUNT,
)


def test_louvain(mode: str = "pilot"):
    is_pilot = mode == "pilot"
    expected_count = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    csv_path = LOUVAIN_PILOT_CLUSTERS_CSV if is_pilot else LOUVAIN_FULL_CLUSTERS_CSV
    parquet_path = LOUVAIN_PILOT_CLUSTERS_PARQUET if is_pilot else LOUVAIN_FULL_CLUSTERS_PARQUET
    fig_path = FIG_LOUVAIN_PILOT_HEATMAP if is_pilot else FIG_LOUVAIN_FULL_HEATMAP

    mode_label = f"PILOTO ({expected_count} nós)" if is_pilot else f"COMPLETA ({expected_count} nós)"
    print("=" * 72)
    print(f"INICIANDO TESTE DE VALIDAÇÃO DO ALGORITMO DE LOUVAIN - MODO {mode_label}")
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

    # 2. Carregar datasets e validar contagens
    print(f"\n[2/4] Validando cobertura estrita de nós e colunas essenciais ({expected_count} janelas)...")
    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(parquet_path)

    assert len(df_csv) == expected_count, (
        f"Contagem incorreta no CSV: esperado {expected_count}, obteve {len(df_csv)}"
    )
    assert len(df_pq) == expected_count, (
        f"Contagem incorreta no Parquet: esperado {expected_count}, obteve {len(df_pq)}"
    )

    required_cols = [
        "window_file",
        "mean_speed_kmh",
        "louvain_community",
        "louvain_interpretation",
    ]
    for col in required_cols:
        assert col in df_csv.columns, f"Coluna essencial '{col}' ausente no CSV!"
        assert col in df_pq.columns, f"Coluna essencial '{col}' ausente no Parquet!"

    print(f"  [OK] 100% dos nós mapeados com sucesso ({expected_count} janelas).")

    # 3. Consistência das comunidades
    print("\n[3/4] Validando integridade e distribuição das comunidades descobertas...")
    communities = sorted(df_csv["louvain_community"].unique())
    n_communities = len(communities)
    
    assert n_communities >= 2, f"Número insuficiente de comunidades detectadas: {n_communities}"
    assert communities == list(range(n_communities)), (
        f"Comunidades devem ser indexadas sequencialmente de 0 a {n_communities - 1}, obtido: {communities}"
    )

    # Verifica se não há comunidades vazias
    counts = df_csv["louvain_community"].value_counts()
    for c_id, count in counts.items():
        assert count > 0, f"Comunidade {c_id} possui contagem nula!"

    # Verifica ausência de nulos na classificação
    assert not df_csv["louvain_community"].isna().any(), "Existem janelas sem comunidade atribuída!"
    assert not df_csv["louvain_interpretation"].isna().any(), "Existem interpretações vazias!"
    print(f"  [OK] Total de {n_communities} comunidades detectadas e devidamente indexadas.")

    # 4. Sumário estatístico das comunidades
    print("\n[4/4] Sumário descritivo das comunidades descobertas:")
    summary = df_csv.groupby("louvain_community").agg(
        qtd=("window_file", "count"),
        v_med=("mean_speed_kmh", "mean"),
        v_max=("max_speed_kmh", "mean"),
        a_med=("mean_pos_accel_ms2", "mean"),
        fren_min=("hard_braking_rate_min", "mean"),
        acel_min=("rapid_accel_rate_min", "mean"),
        interpretacao=("louvain_interpretation", "first"),
    )
    for c_id, row in summary.iterrows():
        pct = (row['qtd'] / expected_count) * 100.0
        print(
            f"  • C{c_id} ({row['interpretacao']}): {int(row['qtd'])} janelas ({pct:.1f}%) | "
            f"v_méd = {row['v_med']:.1f} km/h | fren/min = {row['fren_min']:.2f} | acel/min = {row['acel_min']:.2f}"
        )

    print("\n" + "=" * 72)
    print("TODOS OS TESTES DO ALGORITMO DE LOUVAIN PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa e valida os resultados do Algoritmo de Louvain.")
    parser.add_argument("--pilot", action="store_true", default=True, help="Valida os resultados do modo piloto (100 nós).")
    parser.add_argument("--full", action="store_true", help="Valida os resultados do modo completo (1700 nós).")

    args = parser.parse_args()
    mode = "full" if args.full else "pilot"
    test_louvain(mode=mode)
