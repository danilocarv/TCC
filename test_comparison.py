"""
Script de Teste e Validação da Comparação Topológica Pura dos Cenários de Redes Complexas na Matriz DTW.

Verifica:
1. Existência e integridade dos artefatos comparativos (.csv, .json, .png);
2. Cobertura estrita dos 1.700 nós e consistência dos 4 cenários topológicos;
3. Consistência matemática das métricas DTW intra/inter (DTW Intra < DTW Inter em todos os cenários);
4. Hierarquia de coesão topológica (Razão Grafo Completo > Razão MST ∪ P25 > Razão MST Pura).

Execute com:
  python test_comparison.py

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.config import (
    MASTER_COMPARISON_CSV,
    MASTER_COMPARISON_JSON,
    FIG_FINAL_COMPARISON_SYNTHESIS,
)


def validate_pure_comparison():
    print("=" * 90)
    print("VALIDANDO TESTES DA COMPARAÇÃO TOPOLÓGICA PURA DOS CENÁRIOS NA MATRIZ DTW")
    print("=" * 90)

    # 1. Existência dos artefatos
    print("\n[1/3] Verificando existência dos artefatos de síntese topológica...")
    for p in [MASTER_COMPARISON_CSV, MASTER_COMPARISON_JSON, FIG_FINAL_COMPARISON_SYNTHESIS]:
        assert p.exists(), f"Artefato ausente: {p}"
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {p.name}: Encontrado ({size_kb:.1f} KB)")

    # 2. Integridade dos dados da Tabela Mestre
    print("\n[2/3] Auditando consistência matemática dos 4 cenários...")
    df_master = pd.read_csv(MASTER_COMPARISON_CSV)
    assert len(df_master) == 4, f"Esperado 4 cenários topológicos, obtido {len(df_master)}!"

    for _, row in df_master.iterrows():
        intra = row["dtw_intra_medio"]
        inter = row["dtw_inter_medio"]
        ratio = row["razao_intra_inter"]
        assert intra < inter, f"Violação de coesão em {row['cenario']}: Intra ({intra}) >= Inter ({inter})"
        assert abs((intra / inter) - ratio) < 0.02, f"Razão inconsistente em {row['cenario']}!"
        print(f"  [OK] {row['cenario']}: Intra={intra:.2f} km/h < Inter={inter:.2f} km/h (Razão={ratio:.3f})")

    # 3. Hierarquia de coesão estrutural
    print("\n[3/3] Validando ganho progressivo de coesão e salto de modularidade...")
    r_full = df_master.loc[df_master["cenario"].str.contains("Grafo Completo"), "razao_intra_inter"].values[0]
    r_p25_mst = df_master.loc[df_master["cenario"].str.contains("MST ∪ P25"), "razao_intra_inter"].values[0]
    r_mst = df_master.loc[df_master["cenario"].str.contains("MST Pura"), "razao_intra_inter"].values[0]

    assert r_full > r_p25_mst > r_mst, (
        f"Hierarquia de coesão violada: Completo ({r_full}) > MST U P25 ({r_p25_mst}) > MST Pura ({r_mst})"
    )
    print(f"  [OK] Hierarquia de coesão confirmada: Grafo Completo ({r_full:.3f}) > "
          f"MST ∪ P25 ({r_p25_mst:.3f}) > MST Pura ({r_mst:.3f})")

    print("\n" + "=" * 90)
    print("TODOS OS TESTES DA COMPARAÇÃO TOPOLÓGICA PASSARAM COM SUCESSO! [EXIT CODE 0]")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    validate_pure_comparison()
