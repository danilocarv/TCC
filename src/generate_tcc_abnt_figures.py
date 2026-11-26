"""
Gera as figuras finais do TCC em padrão ABNT otimizadas para leitura direta em folha A4
(largura útil ~16 cm, 300 DPI, fontes ampliadas sem encolhimento excessivo, sem suptitle,
sem menção ao Dummy, vírgula decimal em eixos e rótulos, rótulos em português e p.p. no ΔSOC)
em outputs/figures/tcc_abnt/.

Figuras geradas (sem alterar nem sobrescrever arquivos originais):
- figura_2_kmeans_pca_treino_teste.png       (Seção 4.2 — PCA K-Means 750 vs. 217)
- figura_3_dispersao_3_modelos_teste.png     (Seção 4.3 — Dispersão Real × Predito, 97 viagens)
- figura_4_shap_comparativo_3_modelos.png    (Seção 4.3 — SHAP comparativo 2×1 legível em A4)
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

from src.config import (
    KMEANS_TRAIN_MODEL_PATH,
    KMEANS_TRAIN_TEST_VALIDATION_JSON,
    PCA_TRAIN_MODEL_PATH,
    PROCESSED_TEST_LABELED_PARQUET,
    PROCESSED_TRAIN_LABELED_PARQUET,
)
from src.mlp_consumption_model import (
    MLP_FINAL_CONFIGS,
    _load_train_split,
)
from src.train_test_clustering import DrivingProfileModel  # noqa: F401
from src.xgboost_consumption_model import (
    FEATURE_COLS,
    OUTPUT_MODELS_DIR,
    OUTPUT_RESULTS_DIR,
    TARGET_CONFIGS,
)

ABNT_OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "tcc_abnt"

C_LR = "#EA580C"   # Laranja
C_XGB = "#1D4ED8"  # Azul
C_MLP = "#047857"  # Verde esmeralda

FEATURE_LABELS_PT = {
    "distance_km": "Distância percorrida",
    "duration_s": "Duração da viagem",
    "mean_speed_kmh": "Velocidade média",
    "std_speed_kmh": "Desvio-padrão da velocidade",
    "mean_pos_accel_ms2": "Aceleração positiva média",
    "mean_neg_accel_ms2": "Aceleração negativa média",
    "elevation_gain_m": "Ganho em subidas (SRTM)",
    "elevation_loss_m": "Perda em descidas (SRTM)",
    "net_grade_percent": "Inclinação líquida (SRTM)",
    "aux_hvac_energy_kwh": "Climatização (HVAC)",
    "mean_oat_c": "Temperatura externa",
    "initial_soc_percent": "SOC inicial da bateria",
}


def fmt_pt(val: float, decimals: int = 2) -> str:
    """Formata número no padrão brasileiro (arredondamento ABNT half-up, vírgula decimal)."""
    q = Decimal(10) ** (-decimals)
    d = Decimal(str(val)).quantize(q, rounding=ROUND_HALF_UP)
    s = f"{d:.{decimals}f}".replace(".", ",")
    if s.startswith("-"):
        s = "−" + s[1:]
    return s


def pt_tick_formatter(decimals: int | None = None) -> FuncFormatter:
    """FuncFormatter com vírgula decimal e sinal de menos tipográfico."""
    def _formatter(x: float, _pos: int) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        if decimals is None:
            if float(x).is_integer():
                s = f"{int(round(x))}"
            else:
                s = f"{x:g}".replace(".", ",")
        else:
            s = f"{x:.{decimals}f}".replace(".", ",")
        if s.startswith("-"):
            s = "−" + s[1:]
        return s

    return FuncFormatter(_formatter)


def configure_matplotlib_abnt() -> None:
    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.0,
        "xtick.color": "#1E293B",
        "ytick.color": "#1E293B",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": "#0F172A",
    })


def generate_figura_2_kmeans_pca() -> Path:
    """Figura 2 (Seção 4.2): PCA K-Means 750 vs. 217 — otimizada para A4 (8,6 × 4,0 in)."""
    df_train = pd.read_parquet(PROCESSED_TRAIN_LABELED_PARQUET)
    df_test = pd.read_parquet(PROCESSED_TEST_LABELED_PARQUET)
    model = joblib.load(KMEANS_TRAIN_MODEL_PATH)
    pca = joblib.load(PCA_TRAIN_MODEL_PATH)
    val_meta = json.loads(KMEANS_TRAIN_TEST_VALIDATION_JSON.read_text(encoding="utf-8"))

    train_sil = float(val_meta["metrics"]["silhouette_score"]["train"])
    test_sil = float(val_meta["metrics"]["silhouette_score"]["test"])
    var_pc1 = float(pca.explained_variance_ratio_[0] * 100.0)
    var_pc2 = float(pca.explained_variance_ratio_[1] * 100.0)
    centers_pca = pca.transform(model.cluster_centers_)

    colors = {
        "Econômico / Suave": "#047857",
        "Moderado / Regular": "#EA580C",
        "Agressivo / Dinâmico": "#DC2626",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.15), dpi=300, sharex=True, sharey=True)

    panels = [
        (ax1, df_train, f"A. Treino (N = {len(df_train)}; Silhouette = {fmt_pt(train_sil, 4)})", "Centróides (treino)", 0.56, 26),
        (ax2, df_test, f"B. Teste (N = {len(df_test)}; Silhouette = {fmt_pt(test_sil, 4)})", "Centróides fixados", 0.72, 34),
    ]

    for idx, (ax, df_sub, title, center_lbl, alpha, size) in enumerate(panels):
        for label, color in colors.items():
            sub = df_sub[df_sub["cluster_label"] == label]
            pct = len(sub) / len(df_sub) * 100.0
            short_lbl = label.split(" / ")[0]
            ax.scatter(
                sub["pca_x"],
                sub["pca_y"],
                c=color,
                label=f"{short_lbl} ({len(sub)} | {fmt_pt(pct, 1)}%)",
                alpha=alpha,
                s=size,
                edgecolors="none",
                zorder=3,
            )

        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c="#0F172A",
            s=145,
            marker="X",
            edgecolors="white",
            linewidths=1.4,
            label=center_lbl,
            zorder=10,
        )

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=9, color="#0F172A")
        ax.set_xlabel(f"Componente principal 1 ({fmt_pt(var_pc1, 1)}%)", fontsize=9.8, color="#1E293B")
        if idx == 0:
            ax.set_ylabel(f"Componente principal 2 ({fmt_pt(var_pc2, 1)}%)", fontsize=9.8, color="#1E293B")
        ax.xaxis.set_major_formatter(pt_tick_formatter())
        ax.yaxis.set_major_formatter(pt_tick_formatter())
        ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
        ax.legend(
            loc="upper right", frameon=True, framealpha=0.95,
            facecolor="white", edgecolor="#CBD5E1", fontsize=8.3,
            handletextpad=0.4, borderpad=0.38,
        )
        sns.despine(ax=ax)

    plt.tight_layout(w_pad=1.6)
    out_path = ABNT_OUT_DIR / "figura_2_kmeans_pca_treino_teste.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_figura_3_dispersao_3_modelos() -> Path:
    """Figura 3 (Seção 4.3): Dispersão Real × Predito — otimizada para A4 (8,8 × 4,35 in)."""
    lr_data = json.loads((OUTPUT_RESULTS_DIR / "linear_regression_baseline_evaluation.json").read_text(encoding="utf-8"))
    xgb_data = json.loads((OUTPUT_RESULTS_DIR / "xgboost_dual_model_evaluation.json").read_text(encoding="utf-8"))
    mlp_data = json.loads((OUTPUT_RESULTS_DIR / "mlp_dual_model_evaluation.json").read_text(encoding="utf-8"))

    df_valid, train_idx, test_idx, _, _ = _load_train_split()
    X_train = df_valid[FEATURE_COLS].iloc[train_idx].copy()
    X_test = df_valid[FEATURE_COLS].iloc[test_idx].copy()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.35), dpi=300)

    panels_cfg = [
        (0, "energy_net_kwh", "A. Energia líquida (kWh)", "kWh", -0.15, 1),
        (1, "delta_soc_percent", "B. Queda de carga — ΔSOC (p.p.)", "p.p.", -1.25, 0),
    ]

    for col_idx, target_key, panel_title, unit, y_floor, tick_dec in panels_cfg:
        cfg = TARGET_CONFIGS[target_key]
        y = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)
        y_train, y_test = y[train_idx], y[test_idx]

        lr_model = LinearRegression().fit(X_train, y_train)
        xgb_bundle = joblib.load(OUTPUT_MODELS_DIR / cfg["model_file"])
        mlp_bundle = joblib.load(OUTPUT_MODELS_DIR / MLP_FINAL_CONFIGS[target_key]["model_file"])

        y_lr = lr_model.predict(X_test)
        y_xgb = xgb_bundle["model"].predict(X_test)
        y_mlp = mlp_bundle["model"].predict(X_test)

        ax = axes[col_idx]
        hi = max(float(np.max(y_test)), float(np.max(y_lr)), float(np.max(y_xgb)), float(np.max(y_mlp))) * 1.07

        ax.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0, zorder=1)
        ax.plot([0.0, hi], [0.0, hi], color="#0F172A", linestyle="--", linewidth=1.35, label="Ideal (y = x)", zorder=2)

        ax.scatter(
            y_test, y_lr, facecolors="none", edgecolors=C_LR,
            s=36, linewidths=1.25, alpha=0.86, label="Regressão Linear", zorder=3,
        )
        ax.scatter(
            y_test, y_xgb, marker="^", color=C_XGB, edgecolors="white",
            s=36, linewidths=0.35, alpha=0.84, label="XGBoost", zorder=4,
        )
        ax.scatter(
            y_test, y_mlp, marker="o", color=C_MLP, edgecolors="white",
            s=32, linewidths=0.4, alpha=0.92, label="MLP (16, 8)", zorder=5,
        )

        r2_mlp = mlp_data["models"][target_key]["holdout_test"]["r2"]
        r2_lr = lr_data["models"][target_key]["holdout_test"]["r2"]
        r2_xgb = xgb_data["models"][target_key]["holdout_test"]["r2"]
        r2_box = (
            "R² no teste (N = 97):\n"
            f"MLP (16, 8): {fmt_pt(r2_mlp, 4)}\n"
            f"Linear: {fmt_pt(r2_lr, 4)}\n"
            f"XGBoost: {fmt_pt(r2_xgb, 4)}"
        )
        ax.text(
            0.035, 0.96, r2_box,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.6, color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
            zorder=8,
        )

        ax.set_xlim(-0.04 * hi, hi)
        ax.set_ylim(y_floor, hi)
        ax.set_title(panel_title, fontsize=10.8, fontweight="bold", pad=9, color="#0F172A")
        ax.set_xlabel(f"Valor real ({unit})", fontsize=9.8, color="#1E293B")
        ax.set_ylabel(f"Valor predito ({unit})", fontsize=9.8, color="#1E293B")
        ax.xaxis.set_major_formatter(pt_tick_formatter(tick_dec))
        ax.yaxis.set_major_formatter(pt_tick_formatter(tick_dec))
        ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=4, frameon=True, framealpha=0.96,
        facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.2,
        bbox_to_anchor=(0.5, 1.03),
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.925], w_pad=1.8)
    out_path = ABNT_OUT_DIR / "figura_3_dispersao_3_modelos_teste.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_figura_4_shap_comparativo() -> Path:
    """Figura 4 (Seção 4.3, substitui Tabela 7): SHAP comparativo em 1×2 otimizado para A4."""
    lr_data = json.loads((OUTPUT_RESULTS_DIR / "linear_regression_baseline_evaluation.json").read_text(encoding="utf-8"))
    xgb_df = pd.read_csv(OUTPUT_RESULTS_DIR / "xgboost_shap_rankings.csv")
    mlp_df = pd.read_csv(OUTPUT_RESULTS_DIR / "mlp_shap_rankings.csv")

    # Rótulos concisos para maximizar a área das barras e o tamanho da fonte no layout 1×2 em A4
    short_labels_pt = {
        "distance_km": "Distância",
        "duration_s": "Duração",
        "mean_speed_kmh": "Vel. média",
        "std_speed_kmh": "Desv.-padrão vel.",
        "mean_pos_accel_ms2": "Acel. pos. média",
        "mean_neg_accel_ms2": "Acel. neg. média",
        "elevation_gain_m": "Ganho subidas",
        "elevation_loss_m": "Perda descidas",
        "net_grade_percent": "Inclinação líq.",
        "aux_hvac_energy_kwh": "Climatiz. (HVAC)",
        "mean_oat_c": "Temp. externa",
        "initial_soc_percent": "SOC inicial",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.25), dpi=300)

    panels_cfg = [
        (0, "energy_net_kwh", "A. Energia líquida (kWh)", "kWh", 2, 0.395),
        (1, "delta_soc_percent", "B. Queda de carga — ΔSOC (p.p.)", "p.p.", 1, 1.40),
    ]

    for col_idx, target_key, panel_title, unit, x_tick_dec, xmax in panels_cfg:
        lr_rows = lr_data["models"][target_key]["shap_and_coefficients"]
        lr_map = {r["feature"]: float(r["mean_abs_shap"]) for r in lr_rows}
        lr_sign = {r["feature"]: ("+" if float(r["coef"]) >= 0 else "−") for r in lr_rows}

        xgb_sub = xgb_df[xgb_df["target"] == target_key]
        xgb_map = dict(zip(xgb_sub["feature"], xgb_sub["mean_abs_shap"].astype(float)))

        mlp_sub = mlp_df[mlp_df["target"] == target_key]
        mlp_map = dict(zip(mlp_sub["feature"], mlp_sub["mean_abs_shap"].astype(float)))

        top_union: set[str] = set()
        top_union.update([r["feature"] for r in lr_rows[:6]])
        top_union.update(xgb_sub["feature"].head(6).tolist())
        top_union.update(mlp_sub["feature"].head(6).tolist())

        ordered_feats = sorted(
            top_union,
            key=lambda f: (lr_map[f] + xgb_map[f] + mlp_map[f]) / 3.0,
            reverse=True,
        )

        ax = axes[col_idx]
        y = np.arange(len(ordered_feats))
        h = 0.255

        vals_mlp = [mlp_map[f] for f in ordered_feats]
        vals_xgb = [xgb_map[f] for f in ordered_feats]
        vals_lr = [lr_map[f] for f in ordered_feats]

        b_mlp = ax.barh(y - h, vals_mlp, height=h, color=C_MLP, edgecolor="#0F172A", linewidth=0.55, label="MLP (16, 8) — KernelExplainer", zorder=3)
        b_xgb = ax.barh(y, vals_xgb, height=h, color=C_XGB, edgecolor="#0F172A", linewidth=0.55, label="XGBoost — TreeExplainer", zorder=3)
        b_lr = ax.barh(y + h, vals_lr, height=h, color=C_LR, edgecolor="#0F172A", linewidth=0.55, label="Regressão Linear — LinearExplainer (+ / −)", zorder=3)

        pad_x = xmax * 0.016
        for bar, v in zip(b_mlp, vals_mlp):
            ax.text(v + pad_x, bar.get_y() + bar.get_height() / 2, fmt_pt(v, 4),
                    va="center", ha="left", fontsize=7.9, fontweight="bold", color="#065F46")
        for bar, v in zip(b_xgb, vals_xgb):
            ax.text(v + pad_x, bar.get_y() + bar.get_height() / 2, fmt_pt(v, 4),
                    va="center", ha="left", fontsize=7.8, color="#1E3A8A")
        for bar, v, f in zip(b_lr, vals_lr, ordered_feats):
            ax.text(v + pad_x, bar.get_y() + bar.get_height() / 2, f"{fmt_pt(v, 4)} ({lr_sign[f]})",
                    va="center", ha="left", fontsize=7.8, color="#9A3412")

        for sep in np.arange(0.5, len(ordered_feats) - 0.5, 1.0):
            ax.axhline(sep, color="#E2E8F0", linestyle="-", linewidth=0.8, zorder=1)

        ax.set_yticks(y)
        ax.set_yticklabels([short_labels_pt[f] for f in ordered_feats], fontsize=9.3, color="#0F172A")
        ax.invert_yaxis()
        ax.set_xlim(0.0, xmax)
        ax.xaxis.set_major_formatter(pt_tick_formatter(x_tick_dec))
        ax.set_title(panel_title, fontsize=10.8, fontweight="bold", pad=9, color="#0F172A")
        ax.set_xlabel(f"Impacto médio absoluto |SHAP| ({unit})", fontsize=9.5, color="#1E293B")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=3, frameon=True, framealpha=0.96,
        facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8.8,
        bbox_to_anchor=(0.5, 1.025),
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.935], w_pad=1.6)
    out_path = ABNT_OUT_DIR / "figura_4_shap_comparativo_3_modelos.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    ABNT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_abnt()

    p2 = generate_figura_2_kmeans_pca()
    print(f"[OK] Figura 2 (A4) gerada: {p2}")

    p3 = generate_figura_3_dispersao_3_modelos()
    print(f"[OK] Figura 3 (A4) gerada: {p3}")

    p4 = generate_figura_4_shap_comparativo()
    print(f"[OK] Figura 4 (A4 2x1) gerada: {p4}")


if __name__ == "__main__":
    main()
