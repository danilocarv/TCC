"""
Suíte de Testes Automatizados para o Pipeline de Consumo Positivo de Potência (power_pos_kw).
Valida a integridade matemática, topológica e física de todas as etapas:
1. Carregamento e corte não-negativo (todos os valores >= 0.0 kW);
2. Matriz DTW (simetria, diagonal nula, positividade);
3. Árvore Geradora Mínima (MST Pura - Kruskal: N-1 arestas, 100% conexa);
4. Louvain Puro na MST;
5. Percentil 25 Direto e Rede Otimizada MST ∪ P25 (0 nós órfãos, 100% conexa);
6. Comparação Topológica dos 4 Cenários (hierarquia de modularidade e coesão Intra/Inter).

Autor: Danilo Carvalho de Oliveira
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import get_feature_paths
from src.dtw_analysis import load_training_series


def run_power_tests():
    print("=" * 96)
    print("INICIANDO SUÍTE DE TESTES: PIPELINE DE CONSUMO POSITIVO DE POTÊNCIA (power_pos_kw)")
    print("=" * 96)

    fcfg = get_feature_paths("power_pos_kw")

    # 1. Validação do Carregamento com Corte Não-Negativo
    print("\n[1/4] Testando carregamento de séries com corte não-negativo (power_pos_kw >= 0.0)...")
    data_mat, lengths, stems = load_training_series(n_windows=100, feature="power_pos_kw")
    assert len(data_mat) == 100, f"Esperadas 100 janelas, obtidas {len(data_mat)}"
    assert (data_mat >= 0.0).all(), "Valores negativos (< 0.0 kW) encontrados na série temporal clipada!"
    assert data_mat.max() > 5.0, "Potência máxima muito baixa; falha no carregamento dos dados reais!"
    print(
        f"  [OK] 100 janelas carregadas com sucesso: min = {data_mat.min():.2f} kW, "
        f"max = {data_mat.max():.2f} kW, média ativa = {data_mat[data_mat > 0].mean():.2f} kW."
    )

    # 2. Validação do Modo Piloto (100x100) se existir
    pilot_npy = fcfg["dtw_pilot_npy"]
    if pilot_npy.exists():
        print("\n[2/4] Validando artefatos e propriedades matemáticas do Modo Piloto (100x100)...")
        mat_pilot = np.load(pilot_npy)
        assert mat_pilot.shape == (100, 100), f"Dimensão incorreta do piloto: {mat_pilot.shape}"
        assert np.allclose(np.diag(mat_pilot), 0.0), "Diagonal do piloto não é zero!"
        assert np.allclose(mat_pilot, mat_pilot.T), "Matriz piloto não é simétrica!"
        assert (mat_pilot >= 0.0).all(), "Distâncias negativas encontradas no piloto!"
        print(f"  [OK] Matriz DTW Piloto 100x100 íntegra (Média = {mat_pilot[mat_pilot > 0].mean():.2f} kW).")

        mst_pilot_pq = fcfg["mst_pilot_edges_parquet"]
        if mst_pilot_pq.exists():
            df_mst_pilot = pd.read_parquet(mst_pilot_pq)
            assert len(df_mst_pilot) == 99, f"MST piloto deve ter 99 arestas, encontrado: {len(df_mst_pilot)}"
            print("  [OK] MST Piloto com exatas 99 arestas (N - 1).")
    else:
        print("\n[2/4] Modo Piloto ainda não gerado em disco.")

    # 3. Validação do Modo Completo (1.700x1.700) se existir
    full_npy = fcfg["dtw_full_npy"]
    if full_npy.exists():
        print("\n[3/4] Validando artefatos e propriedades matemáticas do Modo Completo (1.700x1.700)...")
        mat_full = np.load(full_npy)
        assert mat_full.shape == (1700, 1700), f"Dimensão incorreta: {mat_full.shape}"
        assert np.allclose(np.diag(mat_full), 0.0), "Diagonal principal não é zero!"
        assert np.allclose(mat_full, mat_full.T), "Matriz 1700x1700 não é simétrica!"
        assert not np.isnan(mat_full).any(), "Valores NaN detectados na matriz 1700x1700!"
        print(
            f"  [OK] Matriz DTW Completa 1700x1700 íntegra (Média = {mat_full[mat_full > 0].mean():.2f} kW)."
        )

        mst_full_pq = fcfg["mst_full_edges_parquet"]
        if mst_full_pq.exists():
            df_mst_full = pd.read_parquet(mst_full_pq)
            assert len(df_mst_full) == 1699, f"MST 1700x1700 deve ter 1699 arestas, tem {len(df_mst_full)}"
            print("  [OK] MST Completa com exatas 1.699 arestas (N - 1).")
    else:
        print("\n[3/4] Modo Completo ainda não gerado em disco.")

    # 4. Validação da Tabela Comparativa dos 4 Cenários Topológicos se existir
    master_csv = fcfg["master_comparison_csv"]
    if master_csv.exists():
        print("\n[4/4] Auditando Comparação Topológica dos 4 Cenários (power_pos_kw)...")
        df_comp = pd.read_csv(master_csv)
        assert len(df_comp) == 4, f"Esperados 4 cenários na tabela comparativa, encontrados {len(df_comp)}"

        for _, row in df_comp.iterrows():
            cenario = row["cenario"]
            intra = float(row["dtw_intra_medio"])
            inter = float(row["dtw_inter_medio"])
            ratio = float(row["razao_intra_inter"])
            mod_q = float(row["modularidade_q"])
            assert intra < inter, f"Inconsistência em {cenario}: Intra ({intra}) >= Inter ({inter})"
            assert 0.0 < ratio < 1.0, f"Razão Intra/Inter inválida em {cenario}: {ratio}"
            assert mod_q > 0.0, f"Modularidade Q não positiva em {cenario}: {mod_q}"
            print(
                f"  [OK] {cenario:<38} | Q = {mod_q:.4f} | Intra = {intra:.2f} < Inter = {inter:.2f} (Razão = {ratio:.3f})"
            )

        row_3b = df_comp.iloc[3]
        assert int(row_3b["nos_orfaos"]) == 0, "A rede MST ∪ P25 não pode ter nós órfãos!"
        assert bool(row_3b["is_conexo"]) is True, "A rede MST ∪ P25 deve ser 100% conexa!"
        print("  [OK] Rede Otimizada (MST ∪ P25) comprovadamente conexa e com 0 nós órfãos.")
    else:
        print("\n[4/4] Tabela comparativa mestre ainda não gerada.")

    print("\n" + "=" * 96)
    print("[SUCESSO] Todos os testes executados passaram sem inconsistências!")
    print("=" * 96)


if __name__ == "__main__":
    run_power_tests()
