"""
Módulo de Clusterização Não Supervisionada de Perfis de Condução (Etapa 1.3).
Aplica o algoritmo K-Means para agrupar e caracterizar os estilos de condução
(Econômico, Moderado e Agressivo) a partir de variáveis operacionais.

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import os
import sys
import json
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

from src.config import (
    PROCESSED_BEHAVIOR_PARQUET,
    PROCESSED_CLUSTERS_PARQUET,
    PROCESSED_CLUSTERS_CSV,
    KMEANS_MODEL_PATH,
    SCALER_MODEL_PATH,
    PCA_MODEL_PATH,
    CLUSTERING_METRICS_JSON,
    FIG_ELBOW_SILHOUETTE,
    FIG_CLUSTERS_PCA_2D,
    FIG_CLUSTER_PROFILES,
    CLUSTER_FEATURE_COLS,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    ensure_directories,
)


def evaluate_optimal_k(X_scaled: np.ndarray, k_range=range(2, 7)):
    """
    Avalia múltiplos valores de k (número de clusters) utilizando:
    - Inércia (Método do Cotovelo / Elbow Method)
    - Coeficiente de Silhueta (Silhouette Score)
    - Índice Davies-Bouldin
    - Índice Calinski-Harabasz
    Gera o gráfico comparativo acadêmico em alta resolução (300 DPI).
    """
    print("\n-> [1/5] Avaliando número ótimo de clusters k em [2, 6]...")
    metrics = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)

        metrics["k"].append(k)
        metrics["inertia"].append(float(km.inertia_))
        metrics["silhouette"].append(float(sil))
        metrics["davies_bouldin"].append(float(db))
        metrics["calinski_harabasz"].append(float(ch))

        print(
            f"   k={k} | Inércia: {km.inertia_:>9.2f} | Silhueta: {sil:.4f} | "
            f"Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:>7.1f}"
        )

    # Gerar gráfico acadêmico de Cotovelo e Silhueta
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Curva do Cotovelo (Inércia / WCSS)
    ax1.plot(metrics["k"], metrics["inertia"], "o-", color="#1f77b4", linewidth=2.2, markersize=8)
    ax1.set_title("Método do Cotovelo (Inércia / WCSS)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Número de Grupos (k)", fontsize=11)
    ax1.set_ylabel("Inércia Residual (WCSS)", fontsize=11)
    ax1.set_xticks(metrics["k"])
    ax1.axvline(x=3, color="#d62728", linestyle="--", alpha=0.8, label="Ponto Ótimo Selecionado (k=3)")
    ax1.legend(frameon=True, fontsize=10)

    # 2. Coeficiente de Silhueta
    ax2.plot(metrics["k"], metrics["silhouette"], "s-", color="#2ca02c", linewidth=2.2, markersize=8)
    ax2.set_title("Coeficiente Médio de Silhueta (Coesão vs. Separação)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Número de Grupos (k)", fontsize=11)
    ax2.set_ylabel("Score de Silhueta", fontsize=11)
    ax2.set_xticks(metrics["k"])
    ax2.axvline(x=3, color="#d62728", linestyle="--", alpha=0.8, label="Ponto Ótimo Selecionado (k=3)")
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_ELBOW_SILHOUETTE, dpi=300)
    plt.close(fig)
    print(f"   Gráfico de Cotovelo e Silhueta salvo em: {FIG_ELBOW_SILHOUETTE.name}")

    return metrics


class DrivingProfileModel:
    """
    Wrapper do modelo K-Means treinado que garante centróides e rótulos
    semanticamente ordenados por nível de agressividade:
    0: Econômico / Suave
    1: Moderado / Regular
    2: Agressivo / Dinâmico
    """

    def __init__(self, kmeans_model: KMeans, cluster_mapping: dict, label_names: dict):
        self.kmeans = kmeans_model
        self.cluster_mapping = cluster_mapping
        self.label_names = label_names
        self.n_clusters = kmeans_model.n_clusters

        # Centróides ordenados: índice 0=Econômico, 1=Moderado, 2=Agressivo
        inv_map = {v: k for k, v in cluster_mapping.items()}
        self.cluster_centers_ = kmeans_model.cluster_centers_[[inv_map[i] for i in range(self.n_clusters)]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna os IDs ordenados dos clusters (0, 1, 2)."""
        raw_preds = self.kmeans.predict(X)
        return np.array([self.cluster_mapping[r] for r in raw_preds])

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """Retorna os nomes amigáveis dos perfis em texto."""
        ordered_ids = self.predict(X)
        return np.array([self.label_names[i] for i in ordered_ids])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Retorna as distâncias euclidianas a cada centróide ordenado."""
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)


def train_and_label_clusters(
    df: pd.DataFrame,
    X_train_scaled: np.ndarray,
    X_all_scaled: np.ndarray,
    train_mask: np.ndarray,
    n_clusters: int = 3,
):
    """
    Treina o modelo K-Means final com k=3 nos trechos de condução ativa e ordena os centróides:
    - 0: Econômico / Suave
    - 1: Moderado / Regular
    - 2: Agressivo / Dinâmico
    """
    print(f"\n-> [2/5] Treinando modelo K-Means final com k={n_clusters} grupos...")
    km_train = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    train_labels = km_train.fit_predict(X_train_scaled)

    # Calcular centróides dos clusters de treino no espaço original
    df_train = df[train_mask].copy()
    df_train["raw_cluster"] = train_labels

    # Pontuação de agressividade para ordenar semanticamente os grupos:
    # Alta aceleração + alta frenagem brusca + alta oscilação = mais agressivo
    aggressiveness = (
        df_train.groupby("raw_cluster")["hard_braking_rate_min"].mean() * 2.0
        + df_train.groupby("raw_cluster")["rapid_accel_rate_min"].mean() * 2.0
        + df_train.groupby("raw_cluster")["max_pos_accel_ms2"].mean() * 1.0
    ).sort_values()

    # Mapeamento ordenado: 0 = Econômico, 1 = Moderado, 2 = Agressivo
    sorted_raw_ids = aggressiveness.index.tolist()
    cluster_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw_ids)}

    label_names = {
        0: "Econômico / Suave",
        1: "Moderado / Regular",
        2: "Agressivo / Dinâmico",
    }

    # Criar modelo wrapper ordenado
    final_model = DrivingProfileModel(km_train, cluster_mapping, label_names)

    # Atribuir predições a todos os trechos da base
    df["cluster_id"] = final_model.predict(X_all_scaled)
    df["cluster_label"] = final_model.predict_label(X_all_scaled)

    print("   Modelo treinado e centróides ordenados semanticamente com sucesso!")
    return final_model, df


def generate_visualizations(
    df_clusters: pd.DataFrame, X_all_scaled: np.ndarray, kmeans: KMeans
):
    """
    Gera as figuras acadêmicas de alta resolução para o TCC:
    1. Gráfico de dispersão 2D com redução por PCA.
    2. Comparativo dos perfis comportamentais dos centróides.
    """
    print("\n-> [3/5] Gerando visualizações espaciais (PCA) e perfis comparativos...")

    # 1. Redução de Dimensionalidade com PCA (2 componentes)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all_scaled)
    df_clusters["pca_x"] = X_pca[:, 0]
    df_clusters["pca_y"] = X_pca[:, 1]

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    print(
        f"   Variância explicada pelo PCA: PC1={var_pc1:.1f}%, PC2={var_pc2:.1f}% "
        f"(Total: {var_pc1+var_pc2:.1f}%)"
    )

    # Cores padronizadas para cada perfil
    colors = {
        "Econômico / Suave": "#2ca02c",   # Verde
        "Moderado / Regular": "#ff7f0e",  # Amarelo/Laranja
        "Agressivo / Dinâmico": "#d62728", # Vermelho
    }

    # Gráfico de Dispersão PCA 2D
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for label, color in colors.items():
        subset = df_clusters[df_clusters["cluster_label"] == label]
        ax.scatter(
            subset["pca_x"],
            subset["pca_y"],
            c=color,
            label=f"{label} ({len(subset):,d} trechos - {len(subset)/len(df_clusters)*100:.1f}%)",
            alpha=0.55,
            s=40,
            edgecolors="none",
        )

    # Plotar os centróides transformados no espaço PCA
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        s=240,
        marker="X",
        edgecolors="white",
        linewidths=2,
        label="Centróides dos Perfis",
        zorder=10,
    )

    ax.set_title(
        "Espaço Latente de Condução: Projeção 2D dos Clusters de Perfis (PCA)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Componente Principal 1 ({var_pc1:.1f}% variância)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({var_pc2:.1f}% variância)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_CLUSTERS_PCA_2D, dpi=300)
    plt.close(fig)
    print(f"   Gráfico 2D PCA salvo em: {FIG_CLUSTERS_PCA_2D.name}")

    # 2. Gráfico de Comparação de Perfis (Médias por Atributo)
    centroids_original = df_clusters.groupby("cluster_label")[CLUSTER_FEATURE_COLS].mean()

    # Normalizar as métricas de 0 a 1 para visualização em barras comparativas limpas
    centroids_norm = (centroids_original - centroids_original.min()) / (
        centroids_original.max() - centroids_original.min() + 1e-8
    )

    friendly_names = {
        "mean_speed_kmh": "Velocidade Média",
        "std_speed_kmh": "Oscilação Velocidade (Desvio Padrão)",
        "mean_pos_accel_ms2": "Aceleração Média Positiva",
        "max_pos_accel_ms2": "Pico de Aceleração",
        "hard_braking_rate_min": "Taxa Freadas Bruscas/min",
        "rapid_accel_rate_min": "Taxa Arrancadas/min",
    }
    centroids_original.rename(columns=friendly_names, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(CLUSTER_FEATURE_COLS))
    width = 0.26

    labels_order = ["Econômico / Suave", "Moderado / Regular", "Agressivo / Dinâmico"]
    for idx, label in enumerate(labels_order):
        ax.bar(
            x + (idx - 1) * width,
            centroids_norm.loc[label],
            width,
            label=label,
            color=colors[label],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_title(
        "Comparativo Relativo dos Atributos Comportamentais por Perfil de Condução",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [friendly_names[col] for col in CLUSTER_FEATURE_COLS],
        rotation=20,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Intensidade Relativa Normalizada (0 a 1)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 1.18)
    plt.tight_layout()
    fig.savefig(FIG_CLUSTER_PROFILES, dpi=300)
    plt.close(fig)
    print(f"   Gráfico de Perfis Comparativos salvo em: {FIG_CLUSTER_PROFILES.name}")

    return pca, centroids_original


def save_artifacts_and_metrics(
    df_clusters: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    pca: PCA,
    metrics_k: dict,
    centroids_original: pd.DataFrame,
):
    """Salva a base rotulada, os modelos serializados e o JSON de métricas."""
    print("\n-> [4/5] Salvando modelos serializados (.joblib) e base rotulada...")
    ensure_directories()

    # 1. Salvar modelos treinados com joblib
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(scaler, SCALER_MODEL_PATH)
    joblib.dump(pca, PCA_MODEL_PATH)
    print(f"   Modelos serializados salvos em: {MODELS_DIR.name}")

    # 2. Salvar base rotulada em Parquet e CSV
    df_clusters.to_parquet(
        PROCESSED_CLUSTERS_PARQUET, index=False, engine="pyarrow", compression="snappy"
    )
    df_clusters.to_csv(PROCESSED_CLUSTERS_CSV, index=False)
    print(f"   Base rotulada salva em: {PROCESSED_CLUSTERS_PARQUET.name} e .csv")

    # 3. Estatísticas por perfil
    cluster_counts = df_clusters["cluster_label"].value_counts().to_dict()
    total = len(df_clusters)
    cluster_pcts = {k: round(v / total * 100, 2) for k, v in cluster_counts.items()}

    results_data = {
        "evaluation_metrics": metrics_k,
        "optimal_k_selected": 3,
        "cluster_distribution": {
            "counts": cluster_counts,
            "percentages": cluster_pcts,
        },
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "feature_columns_used": CLUSTER_FEATURE_COLS,
    }

    with open(CLUSTERING_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"   Métricas salvas em: {CLUSTERING_METRICS_JSON.name}")

    # 4. Imprimir Relatório Final em Console
    print("\n" + "=" * 72)
    print("-> [5/5] ========== RESULTADOS FINAIS DA CLUSTERIZAÇÃO (K-MEANS) ==========")
    print("=" * 72)
    print("Distribuição dos Trechos da Frota por Perfil de Condução:")
    for label in ["Econômico / Suave", "Moderado / Regular", "Agressivo / Dinâmico"]:
        count = cluster_counts.get(label, 0)
        pct = cluster_pcts.get(label, 0.0)
        print(f"  • {label:<22}: {count:>5,d} trechos ({pct:>5.1f}%)")
    print("-" * 72)
    print("Centróides Médios Reais de Cada Perfil de Condução:")
    for label in ["Econômico / Suave", "Moderado / Regular", "Agressivo / Dinâmico"]:
        c = centroids_original.loc[label]
        print(f"\n  [{label.upper()}]")
        print(f"    - Velocidade Média:         {c['Velocidade Média']:.1f} km/h")
        print(f"    - Oscilação de Velocidade:  {c['Oscilação Velocidade (Desvio Padrão)']:.1f} km/h")
        print(f"    - Aceleração Média Positiva:{c['Aceleração Média Positiva']:.2f} m/s²")
        print(f"    - Pico Médio de Aceleração: {c['Pico de Aceleração']:.2f} m/s²")
        print(f"    - Taxa Freadas Bruscas/min: {c['Taxa Freadas Bruscas/min']:.2f} eventos/min")
        print(f"    - Taxa Arrancadas/min:      {c['Taxa Arrancadas/min']:.2f} eventos/min")
    print("=" * 72 + "\n")


def run_clustering_pipeline():
    """Executa o pipeline completo da Etapa 1.3."""
    if not os.path.exists(PROCESSED_BEHAVIOR_PARQUET):
        raise FileNotFoundError(
            f"Base de atributos não encontrada em: {PROCESSED_BEHAVIOR_PARQUET}. "
            "Execute a Etapa 1.2 primeiro."
        )

    print("-> Carregando base de atributos comportamentais...")
    df = pd.read_parquet(PROCESSED_BEHAVIOR_PARQUET)
    print(f"   Carregados {len(df):,d} trechos operacionais de 120s.")

    # Filtro de condução ativa (vel >= 8.0 km/h) para aprendizado de estilos
    train_mask = (df["mean_speed_kmh"] >= 8.0).values
    df_train = df[train_mask].copy()
    print(f"   Trechos em condução ativa utilizados no treinamento: {len(df_train):,d}")

    # Matrizes de atributos
    X_train = df_train[CLUSTER_FEATURE_COLS].values
    X_all = df[CLUSTER_FEATURE_COLS].values

    # Normalização com StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    # 1. Estudo de k em [2, 6]
    metrics_k = evaluate_optimal_k(X_train_scaled)

    # 2. Treinamento K-Means com k=3
    kmeans, df_labeled = train_and_label_clusters(
        df, X_train_scaled, X_all_scaled, train_mask, n_clusters=3
    )

    # 3. Visualizações PCA e Perfis
    pca, centroids_original = generate_visualizations(df_labeled, X_all_scaled, kmeans)

    # 4. Salvar tudo
    save_artifacts_and_metrics(
        df_labeled, kmeans, scaler, pca, metrics_k, centroids_original
    )


if __name__ == "__main__":
    run_clustering_pipeline()
