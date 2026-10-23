"""
Módulo de Comparação Topológica Final das Abordagens de Redes Complexas sobre a Matriz DTW.

Este módulo consolida a análise comparativa estrutural e topológica estrita (sem rótulos ou classificações prévias):
1. Grafo Completo (1.444.150 arestas, sem filtro);
2. Árvore Geradora Mínima (MST Pura, 1.699 arestas);
3. Percentil 25 Direto (361.038 arestas, 1 nó órfão isolado);
4. Rede Otimizada MST ∪ P25 (361.039 arestas, 100% conexa, 0 órfãos).

Avalia estritamente:
- Arestas retidas e densidade do grafo;
- Conectividade estrita e diagnóstico de nós órfãos;
- Modularidade de Newman (Q);
- Número de comunidades naturais encontradas pelo Louvain puro;
- Coesão temporal na matriz DTW: DTW Intra Médio, DTW Inter Médio e Razão Intra / Inter.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
import argparse
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DTW_ALL_STAGES_COMPARISON_CSV,
    DTW_ALL_STAGES_COMPARISON_JSON,
    MASTER_COMPARISON_CSV,
    MASTER_COMPARISON_JSON,
    FIG_FINAL_COMPARISON_SYNTHESIS,
    FEATURE_REGISTRY,
    get_feature_paths,
    ensure_directories,
)


def run_pure_topological_comparison(feature: str = "speed_kmh") -> pd.DataFrame:
    """
    Consolida o comparativo estritamente topológico dos 4 cenários de rede DTW
    e gera a figura de síntese gráfica sem rótulos comportamentais externos.
    """
    ensure_directories()
    fcfg = get_feature_paths(feature)
    print("=" * 100)
    print(
        f"INICIANDO COMPARAÇÃO TOPOLÓGICA PURA DOS CENÁRIOS NA MATRIZ DTW ({fcfg['name_pt'].upper()})"
    )
    print("=" * 100)

    comp_csv = fcfg["dtw_all_stages_comparison_csv"]
    master_csv = fcfg["master_comparison_csv"]
    master_json = fcfg["master_comparison_json"]
    fig_path = fcfg["fig_final_comparison_synthesis"]

    # Carrega a tabela consolidada das 4 etapas de rede DTW
    if not comp_csv.exists():
        raise FileNotFoundError(f"Tabela comparativa não encontrada em {comp_csv}!")

    df_stages = pd.read_csv(comp_csv)

    # Renomeia colunas para apresentação formal e limpa
    df_stages["modularidade_q"] = df_stages["modularidade_q"].round(4)
    df_stages["pct_arestas"] = df_stages["pct_arestas"].round(2)
    df_stages["dtw_intra_medio"] = df_stages["dtw_intra_medio"].round(2)
    df_stages["dtw_inter_medio"] = df_stages["dtw_inter_medio"].round(2)
    df_stages["razao_intra_inter"] = df_stages["razao_intra_inter"].round(3)

    # Salva versões finais mestre sem rótulos
    df_stages.to_csv(master_csv, index=False, encoding="utf-8")
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(df_stages.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(
        f"\nTABELA COMPARATIVA DOS 4 CENÁRIOS TOPOLÓGICOS ({fcfg['name_pt'].upper()} - LOUVAIN PURO NA MATRIZ DTW):"
    )
    print("=" * 100)
    print(df_stages.to_string(index=False))

    # Gera Figura de Síntese Gráfica Acadêmica em 300 DPI
    plot_pure_topological_synthesis(df_stages, fig_path, feature=feature)

    return df_stages


def plot_pure_topological_synthesis(
    df: pd.DataFrame,
    output_png: Path,
    feature: str = "speed_kmh",
):
    """
    Gera figura em 300 DPI com 3 subplots focados exclusivamente na estrutura da rede:
    1. Densidade de Arestas e Diagnóstico de Conectividade / Nós Órfãos;
    2. Salto da Modularidade de Newman (Q) pelo Louvain Puro;
    3. Coesão na Matriz DTW: Distância Intra vs. Inter e Razão Intra/Inter.
    """
    fcfg = get_feature_paths(feature)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    scenarios = [
        "1. Grafo\nCompleto",
        "2. Árvore\nGeradora (MST)",
        "3A. Percentil 25\nDireto (Sem MST)",
        "3B. Rede Otimizada\n(MST ∪ P25)",
    ]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    # Subplot 1: Modularidade de Newman (Q)
    ax1 = axes[0]
    qs = df["modularidade_q"].values
    bars1 = ax1.bar(scenarios, qs, color=colors, width=0.55, edgecolor="black", linewidth=1.1, alpha=0.9)
    ax1.set_ylabel("Modularidade de Newman (Q)", fontsize=11, fontweight="bold")
    ax1.set_title("Qualidade Comunitária (Modularidade Q)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 1.1)

    for bar, val, n_c in zip(bars1, qs, df["comunidades_total"].values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"Q={val:.4f}\n({n_c} grupos)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Subplot 2: Razão Intra / Inter DTW (Coesão)
    ax2 = axes[1]
    ratios = df["razao_intra_inter"].values
    bars2 = ax2.bar(scenarios, ratios, color=colors, width=0.55, edgecolor="black", linewidth=1.1, alpha=0.9)
    ax2.set_ylabel("Razão DTW Intra / Inter (Menor = Mais Coeso)", fontsize=11, fontweight="bold")
    ax2.set_title("Coesão Interna na Matriz DTW (Razão Intra / Inter)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, max(0.75, float(ratios.max()) * 1.18))

    for bar, val in zip(bars2, ratios):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Subplot 3: Conectividade e Arestas
    ax3 = axes[2]
    edges_str = []
    for idx, row in df.iterrows():
        n_orph = int(row["nos_orfaos"])
        is_conn = bool(row["is_conexo"])
        if n_orph == 0 and is_conn:
            conn_tag = "[0 órfãos - Conexo]"
        elif n_orph == 1:
            conn_tag = "[⚠️ 1 Órfão - Desconexo]"
        else:
            conn_tag = f"[⚠️ {n_orph} Órfãos - Desconexo]"
        edges_str.append(
            f"{int(row['arestas']):,} arestas\n({row['pct_arestas']:.1f}%)\n{conn_tag}"
        )

    bars3 = ax3.bar(scenarios, df["arestas"].values, color=colors, width=0.55, edgecolor="black", linewidth=1.1, alpha=0.9)
    ax3.set_yscale("log")
    ax3.set_ylabel("Quantidade de Arestas no Grafo (Escala Log)", fontsize=11, fontweight="bold")
    ax3.set_title("Volume de Conexões e Diagnóstico de Conectividade", fontsize=12, fontweight="bold", pad=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, label in zip(bars3, edges_str):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.2,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.suptitle(
        f"COMPARAÇÃO ESTRUTURAL DAS ABORDAGENS DE REDES COMPLEXAS NA MATRIZ DTW — {fcfg['name_pt'].upper()} ({fcfg['unit']})",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"\n-> Figura de Síntese Topológica salva em: {output_png.name} ({output_png})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolida a comparação topológica dos 4 cenários sobre a Matriz DTW."
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="speed_kmh",
        choices=list(FEATURE_REGISTRY.keys()),
        help="Feature temporal a analisar (padrão: speed_kmh).",
    )
    args = parser.parse_args()
    run_pure_topological_comparison(feature=args.feature)
