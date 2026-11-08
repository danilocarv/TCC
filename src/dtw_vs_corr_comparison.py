"""
Módulo de Comparação Direta de Contingência: DTW vs. Correlação de Pearson (|r|) em Números Reais.

Gera as 9 matrizes de contingência cruzadas entre as partições do Louvain no DTW
e as partições do Louvain na Correlação de Pearson (3 features x 3 cenários topológicos):
1. Cenário 1: Grafo Completo (DTW vs. Correlação);
2. Cenário 2: Árvore Geradora Mínima - MST (DTW vs. Correlação);
3. Cenário 3: Filtragem por Percentil 25 - P25 (DTW vs. Correlação).

Utiliza valores absolutos (contagem real inteira de janelas de 120s por célula) e calcula:
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- Tabela resumo consolidada (.csv e .json)
- Painel Mestre Figura 29 em 300 DPI (3x3 heatmaps)
"""

import json
import logging
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Configuração de diretórios
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CONTINGENCY_RESULTS_DIR = ROOT_DIR / "outputs" / "results" / "contingency"
FIGURES_KMEANS_DIR = ROOT_DIR / "outputs" / "figures" / "kmeans_baseline"
ARTIFACT_DIR = Path("C:/Users/danil/.gemini/antigravity/brain/32a12848-7a2c-439d-92f4-f6e0db509cff")


def ensure_directories():
    CONTINGENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_KMEANS_DIR.mkdir(parents=True, exist_ok=True)


def plot_single_dtw_vs_corr_ax(
    ax: plt.Axes,
    df_crosstab: pd.DataFrame,
    title: str,
    subtitle: str,
    cmap: str = "Purples"
):
    """
    Plota um único heatmap de contingência com anotação em números reais inteiros.
    """
    k_dtw = df_crosstab.shape[0]
    k_corr = df_crosstab.shape[1]
    matrix_data = df_crosstab.values

    # Determina formato e tamanho de fonte baseado nas dimensões
    is_large = (k_dtw > 12 or k_corr > 12)
    font_size = 7 if is_large else (11 if max(k_dtw, k_corr) <= 5 else 9)

    annot_matrix = np.empty((k_dtw, k_corr), dtype=object)
    for i in range(k_dtw):
        for j in range(k_corr):
            val = int(matrix_data[i, j])
            if val == 0:
                annot_matrix[i, j] = ""
            elif is_large:
                # Na MST (ex: 26x27), mostra números >= 2 para evitar poluição visual
                annot_matrix[i, j] = str(val) if val >= 2 else ""
            else:
                annot_matrix[i, j] = str(val)

    sns.heatmap(
        df_crosstab,
        ax=ax,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        cbar=(not is_large),
        linewidths=0.5 if not is_large else 0.2,
        linecolor="#E2E8F0",
        annot_kws={"size": font_size, "weight": "bold"},
        cbar_kws={"label": "Contagem de Janelas"} if not is_large else None
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Comunidades Louvain DTW", fontsize=10, fontweight="bold")
    ax.set_xlabel("Comunidades Louvain Correlação (|r|)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8 if is_large else 9)


def run_dtw_vs_corr_analysis():
    ensure_directories()
    logger.info("=== INICIANDO COMPARAÇÃO DIRETA DE CONTINGÊNCIA: DTW vs. CORRELAÇÃO (|r|) ===")

    features = ["speed_kmh", "accel_ms2", "power_pos_kw"]
    feature_labels_pt = {
        "speed_kmh": "Velocidade (km/h)",
        "accel_ms2": "Aceleração (m/s²)",
        "power_pos_kw": "Potência Positiva (kW)",
    }

    stages = [
        ("1. Completo", "louvain_clusters", "louvain_community", "corr_louvain_community"),
        ("2. MST", "mst_louvain_clusters", "mst_louvain_community", "corr_mst_community"),
        ("3. P25", "p25_direct_clusters", "p25_direct_louvain_community", "corr_p25_community"),
    ]

    summary_records = []
    panel_items = []  # (f_idx, s_idx, df_ct, title, subtitle)

    comp_counter = 1

    for f_idx, feat in enumerate(features):
        feat_pt = feature_labels_pt[feat]
        feat_dir = PROCESSED_DATA_DIR / feat
        prefix_dtw = "dtw_" if feat == "speed_kmh" else f"dtw_{feat}_"
        prefix_corr = f"corr_{feat}_"

        for s_idx, (stage_name, suffix, dtw_col, corr_col) in enumerate(stages):
            comp_id = f"DVC-{comp_counter:02d}"
            comp_counter += 1

            dtw_file = feat_dir / f"{prefix_dtw}{suffix}_train_750x750.parquet"
            corr_file = feat_dir / f"{prefix_corr}{suffix}_train_750x750.parquet"

            if not dtw_file.exists():
                raise FileNotFoundError(f"Arquivo DTW ausente: {dtw_file}")
            if not corr_file.exists():
                raise FileNotFoundError(f"Arquivo Correlação ausente: {corr_file}")

            df_dtw = pd.read_parquet(dtw_file)
            df_corr = pd.read_parquet(corr_file)

            merged = pd.merge(
                df_dtw[["VehId", "Trip", "window_id", dtw_col]],
                df_corr[["VehId", "Trip", "window_id", corr_col]],
                on=["VehId", "Trip", "window_id"],
            )

            # Filtra nós órfãos (comunidade == -1)
            valid = merged[(merged[dtw_col] != -1) & (merged[corr_col] != -1)].copy()
            orphans_count = len(merged) - len(valid)

            y_dtw = valid[dtw_col].values
            y_corr = valid[corr_col].values

            # Métricas de concordância de partição
            ari = float(adjusted_rand_score(y_dtw, y_corr))
            nmi = float(normalized_mutual_info_score(y_dtw, y_corr))

            # Ordenação consistente dos rótulos
            dtw_unique = sorted(valid[dtw_col].unique())
            corr_unique = sorted(valid[corr_col].unique())
            k_dtw = len(dtw_unique)
            k_corr = len(corr_unique)

            # Tabela de contingência pura (sem margens para o heatmap)
            ct_raw = pd.crosstab(
                valid[dtw_col],
                valid[corr_col],
                rownames=["DTW"],
                colnames=["CORR"]
            ).reindex(index=dtw_unique, columns=corr_unique, fill_value=0)

            # Totais por comunidade para legendas informativas
            dtw_totals = ct_raw.sum(axis=1)
            corr_totals = ct_raw.sum(axis=0)

            df_heatmap = ct_raw.copy()
            df_heatmap.index = [f"DTW C{i} (N={dtw_totals[i]})" for i in dtw_unique]
            df_heatmap.columns = [f"CORR C{j} (N={corr_totals[j]})" for j in corr_unique]

            max_cell = int(ct_raw.values.max())

            # Tabela com margens totais para salvar em CSV
            ct_with_totals = ct_raw.copy()
            ct_with_totals.index = [f"DTW C{i}" for i in dtw_unique]
            ct_with_totals.columns = [f"CORR C{j}" for j in corr_unique]
            ct_with_totals["Total_Linha"] = ct_with_totals.sum(axis=1)
            col_totals = ct_with_totals.sum(axis=0)
            col_totals.name = "Total_Coluna"
            ct_with_totals = pd.concat([ct_with_totals, pd.DataFrame(col_totals).T])

            stage_slug = stage_name.split(". ")[1].lower()
            csv_name = f"{comp_id.lower()}_dtw_vs_corr_{feat}_{stage_slug}_k{k_dtw}x{k_corr}.csv"
            csv_path = CONTINGENCY_RESULTS_DIR / csv_name
            ct_with_totals.to_csv(csv_path, encoding="utf-8-sig")

            title = f"{comp_id}: {feat_pt} - {stage_name}"
            subtitle = f"Matriz {k_dtw}x{k_corr} | ARI = {ari:.4f} | NMI = {nmi:.4f} | Máx = {max_cell}"
            panel_items.append((f_idx, s_idx, df_heatmap, title, subtitle))

            rec = {
                "id": comp_id,
                "feature": feat,
                "feature_pt": feat_pt,
                "cenario": stage_name,
                "k_dtw": int(k_dtw),
                "k_corr": int(k_corr),
                "dimensao_matriz": f"{k_dtw}x{k_corr}",
                "janelas_ativas": int(len(valid)),
                "nos_orfaos": int(orphans_count),
                "total_janelas": int(len(merged)),
                "maior_celula_real": int(max_cell),
                "ari_dtw_vs_corr": round(ari, 4),
                "nmi_dtw_vs_corr": round(nmi, 4),
                "arquivo_csv": csv_name,
            }
            summary_records.append(rec)
            logger.info(f"{comp_id} | {feat_pt:25s} | {stage_name:12s} | {k_dtw:2d}x{k_corr:2d} | N={len(valid)} | ARI={ari:7.4f} | NMI={nmi:7.4f} | MaxCell={max_cell}")

    # Salva resumo consolidado em CSV e JSON
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = CONTINGENCY_RESULTS_DIR / "master_9_dtw_vs_corr_summary.csv"
    summary_json_path = CONTINGENCY_RESULTS_DIR / "master_9_dtw_vs_corr_summary.json"

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Resumo salvo em: {summary_csv_path} e {summary_json_path}")

    # ==========================================================================
    # GERAÇÃO DA FIGURA MESTRE 29: Grade 3x3 de Matrizes de Contingência (300 DPI)
    # ==========================================================================
    logger.info("Gerando Figura Mestre 29: Painel 3x3 DTW vs. Correlação em Números Reais...")
    fig, axes = plt.subplots(3, 3, figsize=(23, 20), dpi=300)

    for r_idx, c_idx, df_ct, t_str, sub_str in panel_items:
        plot_single_dtw_vs_corr_ax(
            ax=axes[r_idx, c_idx],
            df_crosstab=df_ct,
            title=t_str,
            subtitle=sub_str,
            cmap="Purples"
        )

    fig.suptitle(
        "Comparação Direta de Contingência em Números Reais: DTW vs. Correlação de Pearson (|r|)\n"
        "750 Janelas de Treino (120s) - Contagem Absoluta de Viagens por Célula (Completo, MST, P25)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig29_path = FIGURES_KMEANS_DIR / "29_master_dtw_vs_corr_9_contingency_real_counts.png"
    fig.savefig(fig29_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura Mestre 29 salva em: {fig29_path}")

    # Copia para o diretório de artefatos
    if ARTIFACT_DIR.exists():
        dst_fig29 = ARTIFACT_DIR / fig29_path.name
        shutil.copy2(fig29_path, dst_fig29)
        logger.info(f"Copiado para o diretório de artefatos: {dst_fig29}")

    # Imprime tabela síntese no console
    print("\n" + "=" * 105)
    print("SÍNTESE CONSOLIDADA: DTW vs. CORRELAÇÃO DE PEARSON (|r|) - 9 COMPARAÇÕES EM NÚMEROS REAIS")
    print("=" * 105)
    cols_print = ["id", "feature_pt", "cenario", "dimensao_matriz", "janelas_ativas", "maior_celula_real", "ari_dtw_vs_corr", "nmi_dtw_vs_corr"]
    print(summary_df[cols_print].to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    run_dtw_vs_corr_analysis()
