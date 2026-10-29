"""
Módulo de Treinamento Estrito do K-Means (1.700 Treino) e Validação Out-of-Sample (534 Teste).

Executa o particionamento rigoroso sem vazamento de dados (Zero Data Leakage):
1. Fit exclusivo do StandardScaler, PCA e KMeans(k=3, seed=42) nas 1.700 janelas de Treino;
2. Ordenação semântica pura dos centroides (0: Econômico, 1: Moderado, 2: Agressivo);
3. Congelamento dos modelos em disco (.joblib);
4. Transformação e Predição estrita nas 534 janelas inéditas de Teste;
5. Auditoria de generalização out-of-sample:
   - Estabilidade de proporções percentuais;
   - Coeficiente de Silhueta, Inércia média, Davies-Bouldin e Calinski-Harabasz;
   - Consistência de velocidades, acelerações e consumo em kW;
6. Exportação dos datasets rotulados finais (.parquet/.csv) e figuras acadêmicas em 300 DPI.

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
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.config import (
    PROCESSED_TRAIN_FEATURES_PARQUET,
    PROCESSED_TEST_FEATURES_PARQUET,
    PROCESSED_TRAIN_LABELED_PARQUET,
    PROCESSED_TRAIN_LABELED_CSV,
    PROCESSED_TEST_LABELED_PARQUET,
    PROCESSED_TEST_LABELED_CSV,
    KMEANS_TRAIN_MODEL_PATH,
    SCALER_TRAIN_MODEL_PATH,
    PCA_TRAIN_MODEL_PATH,
    KMEANS_TRAIN_TEST_VALIDATION_JSON,
    KMEANS_TRAIN_TEST_VALIDATION_CSV,
    FIG_KMEANS_TRAIN_TEST_PCA,
    FIG_KMEANS_TRAIN_TEST_PROFILES,
    CLUSTER_FEATURE_COLS,
    ensure_directories,
)


class DrivingProfileModel:
    """
    Wrapper do modelo K-Means treinado que garante centroides e rótulos
    semanticamente ordenados por nível de intensidade comportamental:
    0: Econômico / Suave
    1: Moderado / Regular
    2: Agressivo / Dinâmico
    """

    def __init__(self, kmeans_model: KMeans, cluster_mapping: dict, label_names: dict):
        self.kmeans = kmeans_model
        self.cluster_mapping = cluster_mapping
        self.label_names = label_names
        self.n_clusters = kmeans_model.n_clusters

        # Centroides reordenados: índice 0=Econômico, 1=Moderado, 2=Agressivo
        inv_map = {v: k for k, v in cluster_mapping.items()}
        self.cluster_centers_ = kmeans_model.cluster_centers_[[inv_map[i] for i in range(self.n_clusters)]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna os IDs ordenados dos clusters (0, 1, 2)."""
        raw_preds = self.kmeans.predict(X)
        return np.array([self.cluster_mapping[r] for r in raw_preds], dtype=np.int64)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """Retorna os nomes amigáveis dos perfis em texto."""
        ordered_ids = self.predict(X)
        return np.array([self.label_names[i] for i in ordered_ids])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Retorna as distâncias euclidianas a cada centroide ordenado."""
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)


def run_train_test_clustering():
    """Executa o pipeline completo de treino estrito (1.700) e validação (534)."""
    ensure_directories()
    print("=" * 96)
    print("INICIANDO TREINAMENTO DO K-MEANS NO TREINO (1.700) E VALIDAÇÃO NO TESTE (534)")
    print("=" * 96)

    # 1. Carrega dados de Treino e Teste
    if not PROCESSED_TRAIN_FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado em {PROCESSED_TRAIN_FEATURES_PARQUET}!")
    if not PROCESSED_TEST_FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado em {PROCESSED_TEST_FEATURES_PARQUET}!")

    df_train = pd.read_parquet(PROCESSED_TRAIN_FEATURES_PARQUET).copy()
    df_test = pd.read_parquet(PROCESSED_TEST_FEATURES_PARQUET).copy()

    n_train = len(df_train)
    n_test = len(df_test)
    assert n_train == 1700, f"Esperado 1.700 janelas no treino, obtido {n_train}"
    assert n_test == 534, f"Esperado 534 janelas no teste, obtido {n_test}"
    print(f"-> Bases carregadas com sucesso: {n_train:,d} janelas de Treino e {n_test:,d} janelas de Teste.")

    # 2. Fit Exclusivo no Treino (Zero Data Leakage)
    print("\n[1/5] Ajustando StandardScaler, KMeans (k=3) e PCA exclusivamente no Treino...")
    X_train = df_train[CLUSTER_FEATURE_COLS].values
    X_test = df_test[CLUSTER_FEATURE_COLS].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    km_raw = KMeans(n_clusters=3, random_state=42, n_init=20)
    train_raw_labels = km_raw.fit_predict(X_train_scaled)

    # Ordenação semântica pura com base nos centroides de agressividade do treino
    df_train_temp = df_train.copy()
    df_train_temp["raw_cluster"] = train_raw_labels

    aggressiveness = (
        df_train_temp.groupby("raw_cluster")["hard_braking_rate_min"].mean() * 2.0
        + df_train_temp.groupby("raw_cluster")["rapid_accel_rate_min"].mean() * 2.0
        + df_train_temp.groupby("raw_cluster")["max_pos_accel_ms2"].mean() * 1.0
    ).sort_values()

    sorted_raw_ids = aggressiveness.index.tolist()
    cluster_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted_raw_ids)}
    label_names = {
        0: "Econômico / Suave",
        1: "Moderado / Regular",
        2: "Agressivo / Dinâmico",
    }

    model = DrivingProfileModel(km_raw, cluster_mapping, label_names)

    # Ajusta PCA 2D exclusivamente no treino
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Salva modelos congelados
    joblib.dump(model, KMEANS_TRAIN_MODEL_PATH)
    joblib.dump(scaler, SCALER_TRAIN_MODEL_PATH)
    joblib.dump(pca, PCA_TRAIN_MODEL_PATH)
    print(f"  [OK] Modelos congelados salvos em:")
    print(f"       - {KMEANS_TRAIN_MODEL_PATH.name}")
    print(f"       - {SCALER_TRAIN_MODEL_PATH.name}")
    print(f"       - {PCA_TRAIN_MODEL_PATH.name}")

    # 3. Rotulagem do Treino
    df_train["cluster_id"] = model.predict(X_train_scaled)
    df_train["cluster_label"] = model.predict_label(X_train_scaled)
    df_train["pca_x"] = X_train_pca[:, 0]
    df_train["pca_y"] = X_train_pca[:, 1]

    # Distâncias aos centroides
    train_dists_to_centers = model.transform(X_train_scaled)
    for c_id in range(3):
        df_train[f"dist_centroid_{c_id}"] = train_dists_to_centers[:, c_id]

    # 4. Predição Pura no Teste (Out-of-Sample)
    print("\n[2/5] Realizando predição determinística out-of-sample nas 534 janelas de Teste...")
    X_test_scaled = scaler.transform(X_test)  # Transforma usando média/desvio do treino
    test_cluster_ids = model.predict(X_test_scaled)
    test_cluster_labels = model.predict_label(X_test_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # Projeta usando eixos do treino

    df_test["cluster_id"] = test_cluster_ids
    df_test["cluster_label"] = test_cluster_labels
    df_test["pca_x"] = X_test_pca[:, 0]
    df_test["pca_y"] = X_test_pca[:, 1]

    test_dists_to_centers = model.transform(X_test_scaled)
    for c_id in range(3):
        df_test[f"dist_centroid_{c_id}"] = test_dists_to_centers[:, c_id]

    # 5. Exporta Datasets Rotulados
    print("\n[3/5] Exportando bases rotuladas finais (Parquet e CSV)...")
    df_train.to_parquet(PROCESSED_TRAIN_LABELED_PARQUET, index=False, compression="snappy")
    df_train.to_csv(PROCESSED_TRAIN_LABELED_CSV, index=False, encoding="utf-8")
    print(f"  [OK] Treino rotulado salvo: {PROCESSED_TRAIN_LABELED_PARQUET.name} ({n_train} linhas)")

    df_test.to_parquet(PROCESSED_TEST_LABELED_PARQUET, index=False, compression="snappy")
    df_test.to_csv(PROCESSED_TEST_LABELED_CSV, index=False, encoding="utf-8")
    print(f"  [OK] Teste rotulado salvo:  {PROCESSED_TEST_LABELED_PARQUET.name} ({n_test} linhas)")

    # 6. Cálculo das Métricas Comparativas Treino vs. Teste
    print("\n[4/5] Computando métricas de generalização out-of-sample...")
    train_sil = float(silhouette_score(X_train_scaled, df_train["cluster_id"]))
    test_sil = float(silhouette_score(X_test_scaled, df_test["cluster_id"]))

    train_db = float(davies_bouldin_score(X_train_scaled, df_train["cluster_id"]))
    test_db = float(davies_bouldin_score(X_test_scaled, df_test["cluster_id"]))

    train_ch = float(calinski_harabasz_score(X_train_scaled, df_train["cluster_id"]))
    test_ch = float(calinski_harabasz_score(X_test_scaled, df_test["cluster_id"]))

    train_inertia_per_sample = float(km_raw.inertia_ / n_train)
    # Inércia média no teste: menor distância quadrática ao centroide
    min_test_sq_dists = np.min(test_dists_to_centers ** 2, axis=1)
    test_inertia_per_sample = float(np.mean(min_test_sq_dists))

    # Distribuição por Perfil
    train_counts = df_train["cluster_label"].value_counts().to_dict()
    test_counts = df_test["cluster_label"].value_counts().to_dict()

    train_pcts = {k: round(v / n_train * 100.0, 2) for k, v in train_counts.items()}
    test_pcts = {k: round(v / n_test * 100.0, 2) for k, v in test_counts.items()}

    # Médias Cinemáticas e de Potência
    profile_order = ["Econômico / Suave", "Moderado / Regular", "Agressivo / Dinâmico"]
    summary_rows = []

    for prof in profile_order:
        sub_tr = df_train[df_train["cluster_label"] == prof]
        sub_te = df_test[df_test["cluster_label"] == prof]

        summary_rows.append({
            "perfil": prof,
            "treino_count": len(sub_tr),
            "treino_pct": train_pcts.get(prof, 0.0),
            "teste_count": len(sub_te),
            "teste_pct": test_pcts.get(prof, 0.0),
            "diff_pct": round(test_pcts.get(prof, 0.0) - train_pcts.get(prof, 0.0), 2),
            "treino_vel_media_kmh": round(float(sub_tr["mean_speed_kmh"].mean()), 2),
            "teste_vel_media_kmh": round(float(sub_te["mean_speed_kmh"].mean()), 2),
            "treino_acel_max_ms2": round(float(sub_tr["max_pos_accel_ms2"].mean()), 2),
            "teste_acel_max_ms2": round(float(sub_te["max_pos_accel_ms2"].mean()), 2),
            "treino_freadas_min": round(float(sub_tr["hard_braking_rate_min"].mean()), 2),
            "teste_freadas_min": round(float(sub_te["hard_braking_rate_min"].mean()), 2),
            "treino_potencia_kw": round(float(sub_tr["mean_power_kw"].mean()), 2),
            "teste_potencia_kw": round(float(sub_te["mean_power_kw"].mean()), 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(KMEANS_TRAIN_TEST_VALIDATION_CSV, index=False, encoding="utf-8")

    validation_payload = {
        "n_train": n_train,
        "n_test": n_test,
        "features_used": CLUSTER_FEATURE_COLS,
        "metrics": {
            "silhouette_score": {"train": train_sil, "test": test_sil},
            "davies_bouldin": {"train": train_db, "test": test_db},
            "calinski_harabasz": {"train": train_ch, "test": test_ch},
            "inertia_per_sample": {"train": train_inertia_per_sample, "test": test_inertia_per_sample},
        },
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "distribution_summary": summary_rows,
    }

    with open(KMEANS_TRAIN_TEST_VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print("TABELA COMPARATIVA DE GENERALIZAÇÃO: TREINO (1.700) VS. TESTE (534)")
    print("=" * 96)
    print(df_summary[["perfil", "treino_count", "treino_pct", "teste_count", "teste_pct", "diff_pct", "treino_potencia_kw", "teste_potencia_kw"]].to_string(index=False))

    print(f"\n-> Coeficiente de Silhueta: Treino = {train_sil:.4f} | Teste (Out-of-Sample) = {test_sil:.4f}")
    print(f"-> Inércia Média/Amostra:   Treino = {train_inertia_per_sample:.4f} | Teste = {test_inertia_per_sample:.4f}")

    # 7. Gera Visualizações Acadêmicas em 300 DPI
    print("\n[5/5] Gerando figuras acadêmicas em 300 DPI...")
    generate_train_test_visualizations(
        df_train=df_train,
        df_test=df_test,
        model=model,
        pca=pca,
        df_summary=df_summary,
        train_sil=train_sil,
        test_sil=test_sil,
    )

    print("\n" + "=" * 96)
    print("PIPELINE DE TREINAMENTO E VALIDAÇÃO OUT-OF-SAMPLE CONCLUÍDO COM SUCESSO!")
    print("=" * 96 + "\n")

    return df_train, df_test, validation_payload


def generate_train_test_visualizations(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model: DrivingProfileModel,
    pca: PCA,
    df_summary: pd.DataFrame,
    train_sil: float,
    test_sil: float,
):
    """Gera as Figuras 15 e 16 em 300 DPI comparando Treino e Teste."""
    colors = {
        "Econômico / Suave": "#2ca02c",   # Verde
        "Moderado / Regular": "#ff7f0e",  # Laranja
        "Agressivo / Dinâmico": "#d62728", # Vermelho
    }

    # Centroides transformados no espaço PCA
    centers_pca = pca.transform(model.cluster_centers_)

    # -------------------------------------------------------------
    # FIGURA 15: PCA 2D Lado a Lado (Treino 1.700 vs. Teste 534)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharex=True, sharey=True)

    var_pc1 = pca.explained_variance_ratio_[0] * 100.0
    var_pc2 = pca.explained_variance_ratio_[1] * 100.0

    # Painel 1: Treino (1.700)
    for label, color in colors.items():
        sub = df_train[df_train["cluster_label"] == label]
        ax1.scatter(
            sub["pca_x"],
            sub["pca_y"],
            c=color,
            label=f"{label} (N={len(sub):,d} | {len(sub)/len(df_train)*100:.1f}%)",
            alpha=0.55,
            s=36,
            edgecolors="none",
        )

    ax1.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        s=220,
        marker="X",
        edgecolors="white",
        linewidths=1.8,
        label="Centroides Aprendidos (Treino)",
        zorder=10,
    )
    ax1.set_title(
        f"Base de Treino (N={len(df_train):,d} Janelas)\nSilhueta = {train_sil:.4f} (Fit Exclusivo)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel(f"Componente Principal 1 ({var_pc1:.1f}% variância)", fontsize=10.5)
    ax1.set_ylabel(f"Componente Principal 2 ({var_pc2:.1f}% variância)", fontsize=10.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Painel 2: Teste (534)
    for label, color in colors.items():
        sub = df_test[df_test["cluster_label"] == label]
        ax2.scatter(
            sub["pca_x"],
            sub["pca_y"],
            c=color,
            label=f"{label} (N={len(sub):,d} | {len(sub)/len(df_test)*100:.1f}%)",
            alpha=0.65,
            s=42,
            edgecolors="none",
        )

    ax2.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        s=220,
        marker="X",
        edgecolors="white",
        linewidths=1.8,
        label="Centroides Fixados do Treino",
        zorder=10,
    )
    ax2.set_title(
        f"Base de Teste Out-of-Sample (N={len(df_test):,d} Janelas Inéditas)\nSilhueta = {test_sil:.4f} (Predição Pura)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel(f"Componente Principal 1 ({var_pc1:.1f}% variância)", fontsize=10.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(
        "PROJEÇÃO ESPACIAL 2D (PCA): GENERALIZAÇÃO DOS PERFIS K-MEANS EM DADOS INÉDITOS",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_KMEANS_TRAIN_TEST_PCA, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figura 15 (PCA Treino vs. Teste) salva em: {FIG_KMEANS_TRAIN_TEST_PCA.name}")

    # -------------------------------------------------------------
    # FIGURA 16: Comparativo de Atributos Médios (Treino vs. Teste)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300, sharey=False)
    metrics_to_plot = [
        ("Velocidade Média (km/h)", "treino_vel_media_kmh", "teste_vel_media_kmh"),
        ("Taxa Freadas Bruscas (/min)", "treino_freadas_min", "teste_freadas_min"),
        ("Potência Média da Bateria (kW)", "treino_potencia_kw", "teste_potencia_kw"),
    ]

    x = np.arange(len(df_summary))
    width = 0.35

    for ax, (title, tr_col, te_col) in zip(axes, metrics_to_plot):
        bars_tr = ax.bar(x - width/2, df_summary[tr_col], width, label="Treino (1.700)", color="#1f77b4", alpha=0.85, edgecolor="black")
        bars_te = ax.bar(x + width/2, df_summary[te_col], width, label="Teste (534)", color="#2ca02c", alpha=0.85, edgecolor="black")

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split(" / ")[0] for p in df_summary["perfil"]], fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

        # Labels em cima das barras
        for bar in bars_tr:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)
        for bar in bars_te:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle(
        "CONSISTÊNCIA FÍSICA DOS PERFIS: COMPARAÇÃO DIRETA TREINO (1.700) VS. TESTE (534)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_KMEANS_TRAIN_TEST_PROFILES, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figura 16 (Perfis Treino vs. Teste) salva em: {FIG_KMEANS_TRAIN_TEST_PROFILES.name}")


if __name__ == "__main__":
    run_train_test_clustering()
