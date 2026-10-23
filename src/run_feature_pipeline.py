"""
Orquestrador Unificado do Pipeline de Séries Temporais e Redes Complexas por Feature.
Executa sequencialmente toda a cadeia metodológica (DTW -> Grafo Completo Louvain ->
MST Kruskal -> Louvain na MST -> Percentil 25 Direto -> Rede Otimizada MST ∪ P25 ->
Comparação Topológica Síntese) para qualquer feature suportada (ex: accel_ms2, speed_kmh).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import time
import argparse
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FEATURE_REGISTRY, get_feature_paths, ensure_directories
from src.dtw_analysis import run_pilot_dtw, run_full_dtw
from src.louvain_clustering import run_louvain_analysis
from src.mst_builder import run_mst_builder
from src.mst_louvain import run_mst_louvain
from src.percentile_clustering import (
    run_stage_3a_direct_percentile,
    run_stage_3b_mst_percentile,
)
from src.final_comparison import run_pure_topological_comparison


def run_feature_pipeline(feature: str = "accel_ms2", mode: str = "pilot"):
    """
    Executa todas as etapas do pipeline de redes complexas e DTW para a feature escolhida.
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    is_pilot = mode == "pilot"
    mode_label = "PILOTO (100x100)" if is_pilot else "COMPLETO (1.700x1.700)"

    t0 = time.time()
    print("\n" + "#" * 100)
    print(
        f"INICIANDO PIPELINE INTEGRADO DE SÉRIES TEMPORAIS E REDES COMPLEXAS\n"
        f"-> FEATURE: {fcfg['name_pt']} ({fcfg['column']}) [{fcfg['unit']}]\n"
        f"-> MODO:    {mode_label}"
    )
    print("#" * 100 + "\n")

    # 1. Matriz de Similaridade DTW
    print("\n>>> [PASSO 1/6] Calculando Matriz de Distâncias DTW...")
    if is_pilot:
        run_pilot_dtw(feature=feature)
    else:
        run_full_dtw(feature=feature)

    # 2. Louvain Puro no Grafo Completo (Cenário 1)
    print("\n>>> [PASSO 2/6] Executando Louvain Puro no Grafo Completo (Cenário 1)...")
    run_louvain_analysis(is_pilot=is_pilot, feature=feature)

    # 3. Construção da Árvore Geradora Mínima - Kruskal (Cenário 2 - Parte A)
    print("\n>>> [PASSO 3/6] Extraindo Árvore Geradora Mínima - MST Pura (Cenário 2A)...")
    run_mst_builder(is_pilot=is_pilot, feature=feature)

    # 4. Louvain Puro na MST (Cenário 2 - Parte B)
    print("\n>>> [PASSO 4/6] Executando Louvain Puro sobre a MST (Cenário 2B)...")
    run_mst_louvain(is_pilot=is_pilot, feature=feature)

    # 5. Filtragem Percentil 25 Direto (Cenário 3A) e Rede Otimizada MST ∪ P25 (Cenário 3B)
    print("\n>>> [PASSO 5/6] Executando Filtragem Percentil 25 (Cenários 3A e 3B)...")
    run_stage_3a_direct_percentile(is_pilot=is_pilot, feature=feature)
    run_stage_3b_mst_percentile(is_pilot=is_pilot, feature=feature)

    # 6. Síntese Comparativa dos 4 Cenários Topológicos
    print("\n>>> [PASSO 6/6] Consolidando Comparação Topológica dos 4 Cenários...")
    df_comp = run_pure_topological_comparison(feature=feature)

    total_elapsed = time.time() - t0
    print("\n" + "#" * 100)
    print(
        f"[CONCLUÍDO COM SUCESSO] PIPELINE DE {fcfg['name_pt'].upper()} ({mode_label}) "
        f"FINALIZADO EM {total_elapsed:.2f}s!"
    )
    print("#" * 100 + "\n")
    return df_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orquestrador do pipeline completo de DTW e Redes Complexas para uma feature."
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="accel_ms2",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a processar (padrão: accel_ms2).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pilot",
        choices=["pilot", "full"],
        help="Modo de execução: 'pilot' (100x100) ou 'full' (1700x1700).",
    )

    args = parser.parse_args()
    run_feature_pipeline(feature=args.feature, mode=args.mode)
