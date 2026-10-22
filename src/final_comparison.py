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
    ensure_directories,
)


def run_pure_topological_comparison() -> pd.DataFrame:
    """
    Consolida o comparativo estritamente topológico dos 4 cenários de rede DTW
    e gera a figura de síntese gráfica sem rótulos comportamentais externos.
    """
    ensure_directories()
    print("=" * 100)
    print("INICIANDO COMPARAÇÃO TOPOLÓGICA PURA DOS CENÁRIOS DE REDES COMPLEXAS NA MATRIZ DTW")
    print("=" * 100)

    # Carrega a tabela consolidada das 4 etapas de rede DTW
    if not DTW_ALL_STAGES_COMPARISON_CSV.exists():
        raise FileNotFoundError(f"Tabela comparativa não encontrada em {DTW_ALL_STAGES_COMPARISON_CSV}!")

    df_stages = pd.read_csv(DTW_ALL_STAGES_COMPARISON_CSV)

    # Renomeia colunas para apresentação formal e limpa
    df_stages["modularidade_q"] = df_stages["modularidade_q"].round(4)
    df_stages["pct_arestas"] = df_stages["pct_arestas"].round(2)
    df_stages["dtw_intra_medio"] = df_stages["dtw_intra_medio"].round(2)
    df_stages["dtw_inter_medio"] = df_stages["dtw_inter_medio"].round(2)
    df_stages["razao_intra_inter"] = df_stages["razao_intra_inter"].round(3)

    # Salva versões finais mestre sem rótulos
    df_stages.to_csv(MASTER_COMPARISON_CSV, index=False, encoding="utf-8")
    with open(MASTER_COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(df_stages.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print("\nTABELA COMPARATIVA DOS 4 CENÁRIOS TOPOLÓGICOS (LOUVAIN PURO NA MATRIZ DTW):")
    print("=" * 100)
    print(df_stages.to_string(index=False))

    # Gera Figura de Síntese Gráfica Acadêmica em 300 DPI
    plot_pure_topological_synthesis(df_stages, FIG_FINAL_COMPARISON_SYNTHESIS)

    return df_stages


def plot_pure_topological_synthesis(df: pd.DataFrame, output_png: Path):
    """
    Gera figura em 300 DPI com 3 subplots focados exclusivamente na estrutura da rede:
    1. Densidade de Arestas e Diagnóstico de Conectividade / Nós Órfãos;
    2. Salto da Modularidade de Newman (Q) pelo Louvain Puro;
    3. Coesão na Matriz DTW: Distância Intra vs. Inter e Razão Intra/Inter.
    """
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
    ax2.set_ylim(0, 0.75)

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
    edges_str = [f"{int(e):,} arestas\n({p:.1f}%)\n[0 órfãos - Conexo]" for e, p in zip(df["arestas"], df["pct_arestas"])]
    # Ajusta o P25 Direto com o alerta do órfão
    edges_str[2] = f"{int(df.loc[2, 'arestas']):,} arestas\n({df.loc[2, 'pct_arestas']:.1f}%)\n[⚠️ 1 Órfão - Desconexo]"

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
        "COMPARAÇÃO ESTRUTURAL DAS ABORDAGENS DE REDES COMPLEXAS NA MATRIZ DTW (1.700 JANELAS DE TREINO)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"\n-> Figura de Síntese Topológica salva em: {output_png.name} ({output_png})")


if __name__ == "__main__":
    run_pure_topological_comparison()
