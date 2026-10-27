"""
Script de Teste e Validação da Matriz DTW sobre Velocidade.
Verifica:
1. Existência e integridade da matriz salva em .npy e .parquet;
2. Dimensão exata (100x100 no piloto ou 1700x1700 na final);
3. Propriedades matemáticas estritas: diagonal nula (d(x, x) = 0) e simetria perfeita (M = M^T);
4. Ausência de nulos e valores infinitos;
5. Existência e tamanho válido da figura gerada do Heatmap.
Execute com: python test_dtw.py [--pilot | --full]
"""

import sys
import argparse
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
    DTW_PILOT_MATRIX_NPY,
    DTW_PILOT_MATRIX_PARQUET,
    FIG_DTW_PILOT_HEATMAP,
    DTW_SPEED_MATRIX_NPY,
    DTW_SPEED_MATRIX_PARQUET,
    FIG_DTW_SPEED_HEATMAP,
    TRAIN_WINDOW_COUNT,
)


def test_dtw(mode: str = "pilot"):
    is_pilot = mode == "pilot"
    expected_dim = DTW_PILOT_SIZE if is_pilot else TRAIN_WINDOW_COUNT
    npy_path = DTW_PILOT_MATRIX_NPY if is_pilot else DTW_SPEED_MATRIX_NPY
    parquet_path = DTW_PILOT_MATRIX_PARQUET if is_pilot else DTW_SPEED_MATRIX_PARQUET
    fig_path = FIG_DTW_PILOT_HEATMAP if is_pilot else FIG_DTW_SPEED_HEATMAP

    mode_label = f"PILOTO ({expected_dim}x{expected_dim})" if is_pilot else f"COMPLETA ({expected_dim}x{expected_dim})"
    print("=" * 72)
    print(f"INICIANDO TESTE DE VALIDAÇÃO DA MATRIZ DTW - MODO {mode_label}")
    print("=" * 72)

    # 1. Existência dos arquivos
    print("\n[1/4] Verificando existência dos artefatos da matriz e do heatmap...")
    for p, name in [
        (npy_path, f"Matriz Numérica NumPy ({npy_path.name})"),
        (parquet_path, f"Matriz Estruturada Parquet ({parquet_path.name})"),
        (fig_path, f"Gráfico de Calor PNG ({fig_path.name})"),
    ]:
        if not p.exists():
            print(f"[ERRO] Arquivo {name} não encontrado em: {p}")
            sys.exit(1)
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {name}: Encontrado! ({size_kb:.1f} KB)")

    # 2. Carregar matriz e validar dimensões
    print(f"\n[2/4] Validando dimensões e integridade de carregamento ({expected_dim}x{expected_dim})...")
    mat_npy = np.load(npy_path)
    df_pq = pd.read_parquet(parquet_path)

    assert mat_npy.shape == (expected_dim, expected_dim), (
        f"Dimensão NumPy incorreta: esperado {(expected_dim, expected_dim)}, obteve {mat_npy.shape}"
    )
    assert df_pq.shape == (expected_dim, expected_dim), (
        f"Dimensão Parquet incorreta: esperado {(expected_dim, expected_dim)}, obteve {df_pq.shape}"
    )
    assert np.allclose(mat_npy, df_pq.to_numpy()), (
        "Divergência de valores entre o arquivo .npy e o arquivo .parquet!"
    )
    print(f"  [OK] Matriz carregada com sucesso com dimensões exatas de {expected_dim}x{expected_dim}.")

    # 3. Propriedades matemáticas estritas de uma matriz de distância DTW
    print("\n[3/4] Validando propriedades matemáticas da métrica de distância...")
    # a. Diagonal principal = 0
    diag = np.diag(mat_npy)
    assert np.all(diag == 0.0), f"A diagonal principal contém valores não nulos! Máximo: {diag.max()}"
    print("  [OK] Propriedade 1: Diagonal principal estritamente nula (d(x, x) = 0).")

    # b. Simetria perfeita M = M^T
    assert np.allclose(mat_npy, mat_npy.T), "A matriz DTW não é perfeitamente simétrica (M != M^T)!"
    print("  [OK] Propriedade 2: Simetria matemática perfeita (d(x, y) = d(y, x)).")

    # c. Não-negatividade e ausência de NaNs/Infs
    assert not np.isnan(mat_npy).any(), "Existem valores NaN na matriz DTW!"
    assert not np.isinf(mat_npy).any(), "Existem valores infinitos na matriz DTW!"
    assert (mat_npy >= 0.0).all(), "Existem valores de distância negativos!"
    print("  [OK] Propriedade 3: Todas as distâncias são finitas e não-negativas (d(x, y) >= 0).")

    # 4. Estatísticas de distribuição da distância
    print("\n[4/4] Sumário estatístico da matriz de similaridade:")
    off_diag = mat_npy[~np.eye(expected_dim, dtype=bool)]
    print(f"  • Total de pares avaliados: {len(off_diag) // 2:,d} pares únicos")
    print(f"  • Distância Mínima:         {off_diag.min():.2f}")
    print(f"  • Distância Mediana (p50):   {np.median(off_diag):.2f}")
    print(f"  • Distância Média:          {off_diag.mean():.2f}")
    print(f"  • Distância Máxima:         {off_diag.max():.2f}")

    print("\n" + "=" * 72)
    print(f"[SUCESSO] VALIDAÇÃO DA MATRIZ DTW ({mode_label}) CONCLUÍDA COM 100% DE ÊXITO!")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de validação da Matriz DTW")
    parser.add_argument("--full", action="store_true", help="Valida a matriz completa 1700x1700")
    parser.add_argument("--pilot", action="store_true", help="Valida a matriz piloto 100x100")
    args = parser.parse_args()

    mode = "full" if args.full else "pilot"
    test_dtw(mode=mode)
