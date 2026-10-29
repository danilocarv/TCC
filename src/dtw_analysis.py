"""
Módulo de Análise e Cálculo de Similaridade Temporal via Dynamic Time Warping (DTW).
Calcula a matriz de distâncias DTW par-a-par exclusivamente sobre as curvas de velocidade
(speed_kmh) dos arquivos de treino (120s), acelerado via Numba JIT multi-thread.

Suporta:
- Modo Piloto: 100 x 100 janelas (4.950 pares) para validação rápida e inspeção visual;
- Modo Completo: 1.700 x 1.700 janelas (1.444.150 pares) para a base definitiva.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

from src.config import (
    WINDOWS_TRAIN_PARQUET_DIR,
    DTW_PILOT_SIZE,
    DTW_PILOT_MATRIX_NPY,
    DTW_PILOT_MATRIX_PARQUET,
    FIG_DTW_PILOT_HEATMAP,
    DTW_SPEED_MATRIX_NPY,
    DTW_SPEED_MATRIX_PARQUET,
    FIG_DTW_SPEED_HEATMAP,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)


@njit(fastmath=True)
def dtw_distance_single(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Calcula a distância DTW entre duas séries temporais 1D de velocidade
    utilizando programação dinâmica padrão (custo absoluto |v1 - v2|).
    """
    n = len(s1)
    m = len(s2)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        v1 = s1[i - 1]
        for j in range(1, m + 1):
            cost = abs(v1 - s2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[n, m]


@njit(parallel=True, fastmath=True)
def compute_dtw_matrix_numba(data_matrix: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz simétrica de distâncias DTW par-a-par de forma paralela multi-thread.
    data_matrix: array 2D de formato (K, max_len) contendo as velocidades.
    lengths: array 1D com o tamanho real de cada série temporal (30 <= len <= 120).
    """
    k = len(lengths)
    dist_matrix = np.zeros((k, k), dtype=np.float64)

    for i in prange(k):
        n = lengths[i]
        for j in range(i + 1, k):
            m = lengths[j]
            dp = np.full((121, 121), np.inf, dtype=np.float64)
            dp[0, 0] = 0.0

            for t1 in range(1, n + 1):
                v1 = data_matrix[i, t1 - 1]
                for t2 in range(1, m + 1):
                    v2 = data_matrix[j, t2 - 1]
                    cost = abs(v1 - v2)
                    dp[t1, t2] = cost + min(dp[t1 - 1, t2], dp[t1, t2 - 1], dp[t1 - 1, t2 - 1])

            val = dp[n, m]
            dist_matrix[i, j] = val
            dist_matrix[j, i] = val

    return dist_matrix


def load_training_series(
    n_windows: int | None = None,
    feature: str = "speed_kmh",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carrega as séries temporais da feature especificada (ex: Vehicle Speed[km/h], accel_ms2)
    dos arquivos de treino em data/windows/train/parquet/.
    Retorna a matriz de dados (K, 120), o vetor de comprimentos e os nomes dos arquivos.
    """
    fcfg = get_feature_paths(feature)
    col_name = fcfg["column"]

    train_files = sorted(WINDOWS_TRAIN_PARQUET_DIR.glob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(
            f"Nenhum arquivo Parquet encontrado em {WINDOWS_TRAIN_PARQUET_DIR}. "
            "Execute a Etapa 1.2 primeiro."
        )

    if n_windows is not None:
        train_files = train_files[:n_windows]

    k = len(train_files)
    max_len = 120
    data_matrix = np.zeros((k, max_len), dtype=np.float64)
    lengths = np.zeros(k, dtype=np.int64)
    file_stems = []

    clip_positive = fcfg.get("clip_positive", False)

    for idx, fpath in enumerate(train_files):
        df_win = pd.read_parquet(fpath, columns=[col_name])
        vals = df_win[col_name].to_numpy(dtype=np.float64)
        if clip_positive:
            vals = np.maximum(0.0, vals)
        length = len(vals)
        data_matrix[idx, :length] = vals
        lengths[idx] = length
        file_stems.append(fpath.stem)

    return data_matrix, lengths, file_stems


def load_training_speed_series(n_windows: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Alias retrocompatível para carregamento das séries de velocidade."""
    return load_training_series(n_windows=n_windows, feature="speed_kmh")


def plot_dtw_heatmap(
    matrix: np.ndarray,
    output_png: Path,
    title: str,
    subtitle: str | None = None,
    feature: str = "speed_kmh",
):
    """
    Gera e salva o gráfico de calor (Heatmap) em alta resolução (300 DPI)
    representando a matriz de similaridade DTW sobre a feature escolhida.
    """
    fcfg = get_feature_paths(feature)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    k = matrix.shape[0]

    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(matrix, cmap="viridis", aspect="auto", origin="upper", interpolation="nearest")

    cbar = plt.colorbar(pad=0.02)
    cbar.set_label(
        f"Distância DTW Acumulada sobre {fcfg['name_pt']} ({fcfg['unit']})",
        fontsize=11,
        labelpad=10,
    )

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    plt.title(full_title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel(f"Índice da Janela de Treino j (0 a {k - 1})", fontsize=11)
    plt.ylabel(f"Índice da Janela de Treino i (0 a {k - 1})", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap salvo em: {output_png.name} ({output_png})")


def run_pilot_dtw(
    pilot_size: int = DTW_PILOT_SIZE,
    feature: str = "speed_kmh",
) -> np.ndarray:
    """Executa o teste piloto DTW com as primeiras 100 janelas de treino."""
    fcfg = get_feature_paths(feature)
    print("=" * 72)
    print(
        f"INICIANDO EXECUÇÃO PILOTO DTW ({pilot_size}x{pilot_size}) SOBRE {fcfg['name_pt'].upper()} ({fcfg['unit']})"
    )
    print("=" * 72)
    ensure_directories()

    start_t = time.time()
    data_matrix, lengths, file_stems = load_training_series(n_windows=pilot_size, feature=feature)
    print(f"-> Carregadas {len(file_stems)} janelas de treino para o teste piloto ({fcfg['column']}).")

    num_pairs = (pilot_size * (pilot_size - 1)) // 2
    print(f"-> Calculando matriz simétrica de {pilot_size}x{pilot_size} ({num_pairs:,d} pares) com Numba JIT...")

    # Primeira chamada aquece a compilação JIT
    matrix = compute_dtw_matrix_numba(data_matrix, lengths)
    calc_time = time.time() - start_t
    print(f"   Cálculo piloto concluído em {calc_time:.3f}s!")

    # Validações matemáticas do piloto
    assert matrix.shape == (pilot_size, pilot_size), f"Dimensão incorreta: {matrix.shape}"
    assert np.all(np.diag(matrix) == 0.0), "A diagonal principal não é estritamente zero!"
    assert np.allclose(matrix, matrix.T), "A matriz não é perfeitamente simétrica!"
    assert not np.isnan(matrix).any(), "Existem valores NaN na matriz!"
    assert (matrix >= 0.0).all(), "Existem distâncias negativas!"

    print("   [OK] Validação matemática: diagonal nula, simetria M = M^T e distâncias não-negativas.")
    print(f"   Estatísticas DTW do Piloto ({fcfg['name_pt']}):")
    print(f"     • Distância Mínima (fora diagonal): {matrix[matrix > 0].min():.2f} {fcfg['unit']}")
    print(f"     • Distância Média:                 {matrix[matrix > 0].mean():.2f} {fcfg['unit']}")
    print(f"     • Distância Máxima:                {matrix.max():.2f} {fcfg['unit']}")

    # Salvar matriz em disco (.npy e .parquet)
    npy_path = fcfg["dtw_pilot_npy"]
    pq_path = fcfg["dtw_pilot_parquet"]
    fig_path = fcfg["fig_dtw_pilot_heatmap"]

    np.save(npy_path, matrix)
    df_matrix = pd.DataFrame(matrix, index=file_stems, columns=file_stems)
    df_matrix.to_parquet(pq_path, engine="pyarrow", compression="snappy")
    print(f"   Matriz piloto salva em: {npy_path.name} e {pq_path.name}")

    # Gerar o Heatmap piloto
    plot_dtw_heatmap(
        matrix=matrix,
        output_png=fig_path,
        title=f"Mapa de Calor DTW sobre {fcfg['name_pt']} (Piloto {pilot_size}x{pilot_size})",
        subtitle=f"Matriz de Similaridade Temporal de Treino ({num_pairs:,d} pares) | Diagonal Zero",
        feature=feature,
    )

    print("\n" + "=" * 72)
    print(f"[SUCESSO] PILOTO {pilot_size}x{pilot_size} ({fcfg['name_pt']}) CONCLUÍDO E PRONTO PARA AVALIAÇÃO!")
    print("=" * 72)
    return matrix


def run_full_dtw(feature: str = "speed_kmh") -> np.ndarray:
    """Executa a matriz completa DTW 1700x1700 sobre as 1.700 janelas de treino."""
    fcfg = get_feature_paths(feature)
    print("=" * 72)
    print(
        f"INICIANDO EXECUÇÃO COMPLETA DTW (1.700x1.700) SOBRE {fcfg['name_pt'].upper()} ({fcfg['unit']})"
    )
    print("=" * 72)
    ensure_directories()

    start_t = time.time()
    data_matrix, lengths, file_stems = load_training_series(n_windows=None, feature=feature)
    k = len(file_stems)
    num_pairs = (k * (k - 1)) // 2
    print(f"-> Carregadas todas as {k:,d} janelas de treino ({num_pairs:,d} pares únicos) [{fcfg['column']}].")

    print(f"-> Executando cálculo paralelizado com Numba em múltiplos núcleos...")
    matrix = compute_dtw_matrix_numba(data_matrix, lengths)
    calc_time = time.time() - start_t
    print(f"   Cálculo completo concluído em {calc_time:.1f}s!")

    # Validações matemáticas da matriz final
    assert matrix.shape == (k, k), f"Dimensão incorreta: {matrix.shape}"
    assert np.all(np.diag(matrix) == 0.0), "A diagonal principal não é estritamente zero!"
    assert np.allclose(matrix, matrix.T), "A matriz não é perfeitamente simétrica!"
    assert not np.isnan(matrix).any(), "Existem valores NaN na matriz final!"

    # Salvar matriz em disco (.npy e .parquet)
    npy_path = fcfg["dtw_full_npy"]
    pq_path = fcfg["dtw_full_parquet"]
    fig_path = fcfg["fig_dtw_full_heatmap"]

    np.save(npy_path, matrix)
    df_matrix = pd.DataFrame(matrix, index=file_stems, columns=file_stems)
    df_matrix.to_parquet(pq_path, engine="pyarrow", compression="snappy")
    print(f"   Matriz 1700x1700 salva em: {npy_path.name} e {pq_path.name}")

    # Gerar o Heatmap final
    plot_dtw_heatmap(
        matrix=matrix,
        output_png=fig_path,
        title=f"Mapa de Calor DTW sobre {fcfg['name_pt']} (Treino {k}x{k})",
        subtitle=f"Matriz Simétrica de Alinhamento Temporal ({num_pairs:,d} pares) | Semente Shuffle 42",
        feature=feature,
    )

    print("\n" + "=" * 72)
    print(f"[SUCESSO] MATRIZ DTW {k}x{k} ({fcfg['name_pt']}) CALCULADA E HEATMAP GERADO!")
    print("=" * 72)
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cálculo da Matriz DTW sobre Séries Temporais")
    parser.add_argument("--full", action="store_true", help="Executa a matriz completa 1700x1700")
    parser.add_argument("--pilot", action="store_true", help="Executa o teste piloto 100x100")
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a analisar (padrão: speed_kmh)",
    )
    args = parser.parse_args()

    if args.full:
        run_full_dtw(feature=args.feature)
    else:
        run_pilot_dtw(feature=args.feature)
