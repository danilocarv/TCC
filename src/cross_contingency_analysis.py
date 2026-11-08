"""
Módulo de Análise de Contingência Cruzada (18 Comparações: 9 DTW + 9 Correlação vs. K-Means Pareado K x K).

Atende às especificações metodológicas do orientador Prof. Douglas Braz:
1. Avalia os 3 cenários de rede (1. Completo, 2. MST, 3. P25) — excluindo a união (MST ∪ P25);
2. Executa as comparações para as 3 variáveis físicas (speed_kmh, accel_ms2, power_pos_kw):
   - 9 comparações para DTW (3 features x 3 cenários);
   - 9 comparações para Correlação de Pearson (3 features x 3 cenários);
3. Comparação Justa (K x K): Quando o Louvain encontra K != 3 comunidades ativas (ex: K=5, K=6, K=26),
   o K-Means é treinado com exatamente a mesma quantidade de clusters (n_clusters = K) sobre as
   6 features cinemáticas padronizadas das 750 janelas de treino;
4. Exibição em Números Reais (Contagens Absolutas de Janelas) nas tabelas e mapas de calor,
   substituindo porcentagens por valores inteiros exatos.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import (
    PROCESSED_TRAIN_LABELED_PARQUET,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    FIGURES_KMEANS_DIR,
    CLUSTER_FEATURE_COLS,
    get_feature_paths,
    ensure_directories,
)

CONTINGENCY_RESULTS_DIR = RESULTS_DIR / "contingency"


def train_matched_kmeans(
    df_train: pd.DataFrame,
    X_scaled: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Retorna os rótulos do K-Means com exatamente K clusters sobre as 750 janelas de treino.
    - Se k == 3: utiliza o K-Means oficial já rotulado (0: Econômico, 1: Moderado, 2: Agressivo)
      mas também provê ordenação por tamanho ou intensidade. Para manter semântica física clara:
      0: Econômico, 1: Moderado, 2: Agressivo.
    - Se k != 3: ajusta KMeans(n_clusters=k, random_state=42, n_init=20) e ordena os clusters
      por tamanho decrescente (KM0, KM1, ..., KM_{k-1}), espelhando a convenção de ordenação
      das comunidades do Louvain (C0, C1, ..., C_{k-1}).
    """
    if k == 3 and "cluster_id" in df_train.columns:
        labels = df_train["cluster_id"].values.astype(int)
        counts = pd.Series(labels).value_counts().to_dict()
        names = {
            0: f"KM0: Econ.\n(N={counts.get(0, 0)})",
            1: f"KM1: Mod.\n(N={counts.get(1, 0)})",
            2: f"KM2: Agres.\n(N={counts.get(2, 0)})",
        }
        return labels, names

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X_scaled)

    # Ordena clusters do K-Means por tamanho decrescente (em caso de empate, por agressividade)
    vc = pd.Series(raw_labels).value_counts()
    sorted_raw = vc.index.tolist()
    mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw)}
    ordered_labels = np.array([mapping[r] for r in raw_labels], dtype=int)

    counts = pd.Series(ordered_labels).value_counts().to_dict()
    names = {i: f"KM{i}\n(N={counts.get(i, 0)})" for i in range(k)}
    return ordered_labels, names


def extract_louvain_active_partition(
    comm_series: pd.Series,
) -> tuple[np.ndarray, list[int], int, dict[int, int]]:
    """
    Identifica as comunidades ativas (tamanho >= 2) e nós órfãos (tamanho == 1).
    Reindexa as comunidades ativas de 0 a K_ativas - 1 por ordem decrescente de tamanho.
    Nós órfãos recebem rótulo -1.
    """
    vc = comm_series.value_counts()
    active_orig_ids = [int(cid) for cid, cnt in vc.items() if cnt >= 2]
    orphan_orig_ids = [int(cid) for cid, cnt in vc.items() if cnt == 1]

    # Se todas forem tamanho 1 (caso degenerado), considera todas
    if not active_orig_ids:
        active_orig_ids = [int(cid) for cid in vc.index]
        orphan_orig_ids = []

    orig_to_new = {orig_id: new_idx for new_idx, orig_id in enumerate(active_orig_ids)}
    for oid in orphan_orig_ids:
        orig_to_new[oid] = -1

    mapped = np.array([orig_to_new[int(v)] for v in comm_series.values], dtype=int)
    active_counts = {new_idx: int(vc[orig_id]) for new_idx, orig_id in enumerate(active_orig_ids)}
    n_orphans = int(sum(vc[oid] for oid in orphan_orig_ids))

    return mapped, active_orig_ids, n_orphans, active_counts


def build_contingency_tables(
    louvain_mapped: np.ndarray,
    active_counts: dict[int, int],
    n_orphans: int,
    km_labels: np.ndarray,
    km_names: dict[int, str],
    k_active: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói duas versões da tabela de contingência em números reais (contagens absolutas):
    1. df_square (K x K): Apenas comunidades ativas (C0..C_{K-1}) vs. K-Means (KM0..KM_{K-1}).
    2. df_full_margins: Inclui linha de Órfãos (se houver) e margens Total_Linha / Total_Coluna (soma = 750).
    """
    matrix_kxk = np.zeros((k_active, k_active), dtype=int)
    orphan_row = np.zeros(k_active, dtype=int)

    for l_val, km_val in zip(louvain_mapped, km_labels):
        if l_val >= 0:
            matrix_kxk[l_val, km_val] += 1
        else:
            orphan_row[km_val] += 1

    row_labels = [f"C{i} (N={active_counts[i]})" for i in range(k_active)]
    col_labels = [km_names[j].replace("\n", " ") for j in range(k_active)]

    df_square = pd.DataFrame(matrix_kxk, index=row_labels, columns=col_labels)
    df_square.index.name = "Louvain \\ K-Means"

    # Tabela completa com órfãos (se existirem) e totais marginais
    df_full = df_square.copy()
    if n_orphans > 0:
        df_full.loc[f"Órfãos (N={n_orphans})"] = orphan_row

    df_full["Total_Linha"] = df_full.sum(axis=1)
    col_totals = df_full.sum(axis=0)
    df_full.loc["Total_Coluna"] = col_totals

    assert int(df_full.loc["Total_Coluna", "Total_Linha"]) == len(louvain_mapped), (
        f"Erro de balanço de massa! Esperado {len(louvain_mapped)}, "
        f"obtido {df_full.loc['Total_Coluna', 'Total_Linha']}"
    )

    return df_square, df_full


def plot_single_contingency_ax(
    ax: plt.Axes,
    df_square: pd.DataFrame,
    title: str,
    subtitle: str,
    cmap: str = "Blues",
) -> None:
    """
    Renderiza um heatmap de contingência K x K em números inteiros reais em um eixo Matplotlib.
    """
    mat = df_square.values
    k = mat.shape[0]

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=max(1, int(mat.max())))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Nº de Janelas (Real)", fontsize=8)

    # Ajusta tamanho da fonte de acordo com a dimensão K x K
    if k <= 4:
        cell_fs = 11.0
        tick_fs = 8.5
    elif k <= 7:
        cell_fs = 9.0
        tick_fs = 7.5
    else:
        # Caso MST (K ~ 25 a 28)
        cell_fs = 5.2
        tick_fs = 5.5

    thresh = mat.max() / 2.0
    for i in range(k):
        for j in range(k):
            val = int(mat[i, j])
            # Para matrizes grandes (K > 15), destaca valores > 0 com nitidez
            color = "white" if val > thresh else ("#1a1a1a" if val > 0 else "#b0b0b0")
            weight = "bold" if (val > 0 and k <= 10) or (val >= 3 and k > 10) else "normal"
            ax.text(
                j,
                i,
                f"{val:d}",
                ha="center",
                va="center",
                color=color,
                fontsize=cell_fs,
                fontweight=weight,
            )

    if k <= 10:
        x_labels = [c.replace(" (N=", "\n(N=") for c in df_square.columns]
        y_labels = list(df_square.index)
        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels, fontsize=tick_fs, rotation=0)
        ax.set_yticks(range(k))
        ax.set_yticklabels(y_labels, fontsize=tick_fs)
    else:
        # Rótulos compactos para K >= 20 (MST)
        x_labels = [f"KM{j}" for j in range(k)]
        y_labels = [f"C{i}" for i in range(k)]
        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels, fontsize=tick_fs, rotation=90)
        ax.set_yticks(range(k))
        ax.set_yticklabels(y_labels, fontsize=tick_fs)

    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, fontweight="bold", pad=8)
    ax.set_xlabel(f"Clusters K-Means (K={k})", fontsize=8.5, fontweight="bold")
    ax.set_ylabel(f"Comunidades Louvain (K={k})", fontsize=8.5, fontweight="bold")


def run_all_18_comparisons() -> pd.DataFrame:
    """
    Executa as 18 comparações (9 DTW + 9 Correlação) e gera todas as tabelas e figuras em números reais.
    """
    ensure_directories()
    CONTINGENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("INICIANDO ANÁLISE DE CONTINGÊNCIA CRUZADA (18 COMPARAÇÕES: 9 DTW + 9 CORRELAÇÃO vs. K-MEANS PAREADO KxK)")
    print("=" * 105)

    # Carrega dataset de treino rotulado pelo K-Means Baseline (K=3)
    df_train = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET).copy()
    n_windows = len(df_train)
    assert n_windows == 750, f"Esperado 750 janelas de treino, encontrado {n_windows}"

    # Matriz de features padronizadas para treino dinâmico do K-Means quando K != 3
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[CLUSTER_FEATURE_COLS].values)
    km_k3_labels = df_train["cluster_id"].values.astype(int)

    features = ["speed_kmh", "accel_ms2", "power_pos_kw"]
    feature_labels_pt = {
        "speed_kmh": "Velocidade (km/h)",
        "accel_ms2": "Aceleração (m/s²)",
        "power_pos_kw": "Potência Positiva (kW)",
    }

    summary_records = []
    dtw_panels = []   # Lista de (row_idx, col_idx, df_square, title, subtitle)
    corr_panels = []  # Lista de (row_idx, col_idx, df_square, title, subtitle)

    dtw_counter = 1
    corr_counter = 1

    for f_idx, feat in enumerate(features):
        fcfg = get_feature_paths(feat)
        feat_pt = feature_labels_pt[feat]
        prefix_dtw = "dtw_" if feat == "speed_kmh" else f"dtw_{feat}_"
        feat_dir = PROCESSED_DATA_DIR / feat

        # Definição dos 3 cenários para DTW e 3 cenários para Correlação (excluindo MST ∪ P25)
        method_specs = [
            # DTW
            (
                "DTW",
                0,
                "1. Completo",
                feat_dir / f"{prefix_dtw}louvain_clusters_train_750x750.parquet",
                "louvain_community",
            ),
            (
                "DTW",
                1,
                "2. MST",
                feat_dir / f"{prefix_dtw}mst_louvain_clusters_train_750x750.parquet",
                "mst_louvain_community",
            ),
            (
                "DTW",
                2,
                "3. P25",
                feat_dir / f"{prefix_dtw}p25_direct_clusters_train_750x750.parquet",
                "p25_direct_louvain_community",
            ),
            # Correlação
            (
                "Correlação",
                0,
                "1. Completo",
                feat_dir / f"corr_{feat}_louvain_clusters_train_750x750.parquet",
                "corr_louvain_community",
            ),
            (
                "Correlação",
                1,
                "2. MST",
                feat_dir / f"corr_{feat}_mst_louvain_clusters_train_750x750.parquet",
                "corr_mst_community",
            ),
            (
                "Correlação",
                2,
                "3. P25",
                feat_dir / f"corr_{feat}_p25_direct_clusters_train_750x750.parquet",
                "corr_p25_community",
            ),
        ]

        for family, s_idx, stage_name, parquet_path, col_name in method_specs:
            if not parquet_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {parquet_path}")

            df_comm = pd.read_parquet(parquet_path)
            raw_comm = df_comm[col_name]

            # Extrai comunidades ativas (K) e órfãos
            louvain_mapped, active_ids, n_orphans, active_counts = extract_louvain_active_partition(raw_comm)
            k_active = len(active_ids)

            # Treina ou recupera K-Means com exatamente K = k_active clusters
            km_matched_labels, km_names = train_matched_kmeans(df_train, X_scaled, k=k_active)

            # Constrói matrizes de contingência em números reais
            df_square, df_full = build_contingency_tables(
                louvain_mapped=louvain_mapped,
                active_counts=active_counts,
                n_orphans=n_orphans,
                km_labels=km_matched_labels,
                km_names=km_names,
                k_active=k_active,
            )

            # Métricas de concordância (apenas sobre nós ativos e sobre todos os 750 nós)
            active_mask = louvain_mapped >= 0
            ari_matched = float(adjusted_rand_score(km_matched_labels[active_mask], louvain_mapped[active_mask]))
            nmi_matched = float(normalized_mutual_info_score(km_matched_labels[active_mask], louvain_mapped[active_mask]))

            ari_k3 = float(adjusted_rand_score(km_k3_labels, raw_comm.values))
            nmi_k3 = float(normalized_mutual_info_score(km_k3_labels, raw_comm.values))

            # Identificador e salvamento individual
            if family == "DTW":
                comp_id = f"DTW-{dtw_counter:02d}"
                dtw_counter += 1
            else:
                comp_id = f"CORR-{corr_counter:02d}"
                corr_counter += 1

            stage_slug = stage_name.split(". ")[1].lower()
            fam_slug = "dtw" if family == "DTW" else "corr"
            csv_name = f"{comp_id.lower()}_{fam_slug}_{feat}_{stage_slug}_kmeans_k{k_active}x{k_active}.csv"
            csv_path = CONTINGENCY_RESULTS_DIR / csv_name
            df_full.to_csv(csv_path, encoding="utf-8")

            # Soma da diagonal principal (após pareamento por tamanho) e maior célula
            diag_sum = int(np.trace(df_square.values))
            max_cell = int(df_square.values.max())
            active_windows = int(df_square.values.sum())

            summary_records.append({
                "id": comp_id,
                "familia": family,
                "feature": feat,
                "feature_pt": feat_pt,
                "cenario": stage_name,
                "k_louvain_ativas": k_active,
                "k_kmeans_pareado": k_active,
                "dimensao_matriz": f"{k_active}x{k_active}",
                "janelas_ativas": active_windows,
                "nos_orfaos": n_orphans,
                "total_janelas": active_windows + n_orphans,
                "soma_diagonal_principal": diag_sum,
                "maior_celula_real": max_cell,
                "ari_pareado_kxk": round(ari_matched, 4),
                "nmi_pareado_kxk": round(nmi_matched, 4),
                "ari_baseline_k3": round(ari_k3, 4),
                "nmi_baseline_k3": round(nmi_k3, 4),
                "arquivo_csv": csv_name,
            })

            orphan_txt = f" | Órfãos={n_orphans}" if n_orphans > 0 else ""
            title = f"[{comp_id}] {feat_pt} — {stage_name}"
            subtitle = (
                f"Matriz {k_active}x{k_active} (N={active_windows}{orphan_txt}) | "
                f"ARI={ari_matched:.4f} | NMI={nmi_matched:.4f}"
            )

            if family == "DTW":
                dtw_panels.append((f_idx, s_idx, df_square, title, subtitle))
            else:
                corr_panels.append((f_idx, s_idx, df_square, title, subtitle))

            print(f"\n--- {title} ({subtitle}) ---")
            if k_active <= 10:
                print(df_full.to_string())
            else:
                print(f"Matriz {k_active}x{k_active} gerada (salva em {csv_name}). Top 5 linhas x 5 colunas:")
                print(df_full.iloc[:5, :5].to_string())

    # Salva resumo mestre das 18 comparações
    df_summary = pd.DataFrame(summary_records)
    master_csv = CONTINGENCY_RESULTS_DIR / "master_18_contingency_summary.csv"
    master_json = CONTINGENCY_RESULTS_DIR / "master_18_contingency_summary.json"
    df_summary.to_csv(master_csv, index=False, encoding="utf-8")
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # FIGURA MESTRE 1: 9 Comparações do DTW (Grade 3x3 em 300 DPI)
    # --------------------------------------------------------------------------
    fig_dtw, axes_dtw = plt.subplots(3, 3, figsize=(22, 19), dpi=300)
    for r_idx, c_idx, df_sq, t_str, sub_str in dtw_panels:
        plot_single_contingency_ax(
            ax=axes_dtw[r_idx, c_idx],
            df_square=df_sq,
            title=t_str,
            subtitle=sub_str,
            cmap="Blues",
        )
    fig_dtw.suptitle(
        "Comparação de Contingência K x K em Números Reais: Redes DTW (Completo, MST, P25) vs. K-Means Pareado\n"
        "750 Janelas de Treino (120s) — Contagem Absoluta de Viagens por Célula",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig_dtw.tight_layout(rect=[0, 0, 1, 0.98])
    fig_dtw_path = FIGURES_KMEANS_DIR / "23_master_dtw_9_contingency_real_counts.png"
    fig_dtw_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dtw.savefig(fig_dtw_path, bbox_inches="tight")
    plt.close(fig_dtw)
    print(f"\n[OK] Painel Mestre DTW (9 comparações) salvo em: {fig_dtw_path}")

    # --------------------------------------------------------------------------
    # FIGURA MESTRE 2: 9 Comparações da Correlação (Grade 3x3 em 300 DPI)
    # --------------------------------------------------------------------------
    fig_corr, axes_corr = plt.subplots(3, 3, figsize=(22, 19), dpi=300)
    for r_idx, c_idx, df_sq, t_str, sub_str in corr_panels:
        plot_single_contingency_ax(
            ax=axes_corr[r_idx, c_idx],
            df_square=df_sq,
            title=t_str,
            subtitle=sub_str,
            cmap="Oranges",
        )
    fig_corr.suptitle(
        "Comparação de Contingência K x K em Números Reais: Correlação de Pearson (Completo, MST, P25) vs. K-Means Pareado\n"
        "750 Janelas de Treino (120s) — Contagem Absoluta de Viagens por Célula (Similaridade S = |r|)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig_corr.tight_layout(rect=[0, 0, 1, 0.98])
    fig_corr_path = FIGURES_KMEANS_DIR / "24_master_corr_9_contingency_real_counts.png"
    fig_corr.savefig(fig_corr_path, bbox_inches="tight")
    plt.close(fig_corr)
    print(f"[OK] Painel Mestre Correlação (9 comparações) salvo em: {fig_corr_path}")

    print("\n" + "=" * 105)
    print("RESUMO CONSOLIDADO DAS 18 COMPARAÇÕES (9 DTW + 9 CORRELAÇÃO vs. K-MEANS PAREADO KxK):")
    print("=" * 105)
    cols_show = [
        "id", "familia", "feature", "cenario", "dimensao_matriz",
        "janelas_ativas", "nos_orfaos", "maior_celula_real",
        "ari_pareado_kxk", "nmi_pareado_kxk",
    ]
    print(df_summary[cols_show].to_string(index=False))
    print("=" * 105)

    return df_summary


if __name__ == "__main__":
    run_all_18_comparisons()
