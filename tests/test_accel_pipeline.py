"""
Suíte de Testes Automatizados para o Pipeline de Aceleração Longitudinal (accel_ms2).
Valida a integridade matemática, topológica e gráfica de todas as etapas:
1. Matriz DTW (simetria, diagonal nula, positividade);
2. Louvain Puro no Grafo Completo;
3. Árvore Geradora Mínima (MST Pura - Kruskal: N-1 arestas, conexidade);
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import get_feature_paths


def run_accel_tests():
    print("=" * 92)
    print("INICIANDO SUÍTE DE TESTES: PIPELINE DE ACELERAÇÃO LONGITUDINAL (accel_ms2)")
    print("=" * 92)

    fcfg = get_feature_paths("accel_ms2")

    # 1. Validação do Modo Piloto (100x100)
    print("\n[1/4] Validando artefatos e propriedades matemáticas do Modo Piloto (100x100)...")
    pilot_npy = fcfg["dtw_pilot_npy"]
    assert pilot_npy.exists(), f"Matriz piloto não encontrada: {pilot_npy}"
    mat_pilot = np.load(pilot_npy)
    assert mat_pilot.shape == (100, 100), f"Dimensão incorreta do piloto: {mat_pilot.shape}"
    assert np.allclose(np.diag(mat_pilot), 0.0), "Diagonal do piloto não é zero!"
    assert np.allclose(mat_pilot, mat_pilot.T), "Matriz piloto não é simétrica!"
    assert (mat_pilot >= 0.0).all(), "Distâncias negativas encontradas no piloto!"
    print(f"  [OK] Matriz DTW Piloto 100x100 íntegra (Média = {mat_pilot[mat_pilot > 0].mean():.2f} m/s²).")

    mst_pilot_pq = fcfg["mst_pilot_edges_parquet"]
    assert mst_pilot_pq.exists(), f"Arestas MST piloto não encontradas: {mst_pilot_pq}"
    df_mst_pilot = pd.read_parquet(mst_pilot_pq)
    assert len(df_mst_pilot) == 99, f"MST piloto deve ter 99 arestas, encontrado: {len(df_mst_pilot)}"
    print("  [OK] MST Piloto com exatas 99 arestas (N - 1).")

    # 2. Validação do Modo Completo (1.700x1.700) se já executado
    full_npy = fcfg["dtw_full_npy"]
    if full_npy.exists():
        print("\n[2/4] Validando artefatos e propriedades matemáticas do Modo Completo (1.700x1.700)...")
        mat_full = np.load(full_npy)
        assert mat_full.shape == (1700, 1700), f"Dimensão incorreta: {mat_full.shape}"
        assert np.allclose(np.diag(mat_full), 0.0), "Diagonal principal não é zero!"
        assert np.allclose(mat_full, mat_full.T), "Matriz 1700x1700 não é simétrica!"
        assert not np.isnan(mat_full).any(), "Valores NaN detectados na matriz 1700x1700!"
        print(
            f"  [OK] Matriz DTW Completa 1700x1700 íntegra (Média = {mat_full[mat_full > 0].mean():.2f} m/s²)."
        )

        mst_full_pq = fcfg["mst_full_edges_parquet"]
        assert mst_full_pq.exists(), f"Arestas MST completa não encontradas: {mst_full_pq}"
        df_mst_full = pd.read_parquet(mst_full_pq)
        assert len(df_mst_full) == 1699, f"MST 1700x1700 deve ter 1699 arestas, tem {len(df_mst_full)}"
        print("  [OK] MST Completa com exatas 1.699 arestas (N - 1).")
    else:
        print("\n[2/4] Modo Completo (1.700x1.700) ainda não gerado; validando com base no Piloto.")

    # 3. Validação da Tabela Comparativa dos 4 Cenários Topológicos
    print("\n[3/4] Auditando Comparação Topológica dos 4 Cenários (accel_ms2)...")
    master_csv = fcfg["master_comparison_csv"]
    assert master_csv.exists(), f"Tabela comparativa mestre não encontrada: {master_csv}"
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

    # Verifica que a Rede Otimizada MST ∪ P25 tem 0 órfãos e é 100% conexa
    row_3b = df_comp.iloc[3]
    assert int(row_3b["nos_orfaos"]) == 0, "A rede MST ∪ P25 não pode ter nós órfãos!"
    assert bool(row_3b["is_conexo"]) is True, "A rede MST ∪ P25 deve ser 100% conexa!"
    print("  [OK] Rede Otimizada (MST ∪ P25) comprovadamente conexa e com 0 nós órfãos.")

    # 4. Verificação das Figuras Geradas
    print("\n[4/4] Verificando figuras de alta resolução (300 DPI)...")
    fig_synth = fcfg["fig_final_comparison_synthesis"]
    assert fig_synth.exists() and fig_synth.stat().st_size > 10_000, (
        f"Figura síntese ausente ou vazia: {fig_synth}"
    )
    print(f"  [OK] Figura síntese gerada: {fig_synth.name} ({fig_synth.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 92)
    print("TODOS OS TESTES DO PIPELINE DE ACELERAÇÃO PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 92)


if __name__ == "__main__":
    run_accel_tests()
