"""
Módulo de Modelagem e Predição de Consumo Energético com XGBoost Regressor Dual.

Executa a predição supervisionada para viagens veiculares completas (N = 486 viagens válidas)
de veículos 100% elétricos (Nissan Leaf 24 kWh, dataset VED), abordando dois alvos fundamentais:
1. Alvo 1: Energia Líquida Consumida da Bateria (energy_net_kwh em kWh);
2. Alvo 2: Variação Real do Estado de Carga da Bateria (delta_soc_percent em %).

Utiliza um conjunto enxuto de 12 features físicas (cinemática, topografia SRTM, clima e eletroquímica),
validação cruzada estratificada em 5 folds (80% treino), avaliação em teste holdout (20%),
comparação contra baselines (Dummy e Regressão Linear) e explicabilidade via SHAP (TreeExplainer).

Autor: Danilo Carvalho de Oliveira
Orientador: Douglas Donizeti de Castilho Braz
"""

import sys
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Diretórios
DATA_PATH = Path("data/processed/trips/ev_trips_complete_dataset.parquet")
OUTPUT_MODELS_DIR = Path("outputs/models")
OUTPUT_RESULTS_DIR = Path("outputs/results/trips")
OUTPUT_FIGS_DIR = Path("outputs/figures/trips")
ARTIFACT_DIR = Path("C:/Users/danil/.gemini/antigravity/brain/32a12848-7a2c-439d-92f4-f6e0db509cff")

# 12 Features Físicas Selecionadas
FEATURE_COLS = [
    # Cinemática e Operação (6)
    "distance_km",
    "duration_s",
    "mean_speed_kmh",
    "std_speed_kmh",
    "mean_pos_accel_ms2",
    "mean_neg_accel_ms2",

    # Topografia e Altimetria SRTM (3)
    "elevation_gain_m",
    "elevation_loss_m",
    "net_grade_percent",

    # Clima e Conforto Térmico (2)
    "aux_hvac_energy_kwh",
    "mean_oat_c",

    # Eletroquímica da Bateria (1)
    "initial_soc_percent"
]

FEATURE_GROUPS = {
    "distance_km": ("Cinemática & Operação", "#1E40AF"),
    "duration_s": ("Cinemática & Operação", "#1E40AF"),
    "mean_speed_kmh": ("Cinemática & Operação", "#1E40AF"),
    "std_speed_kmh": ("Cinemática & Operação", "#1E40AF"),
    "mean_pos_accel_ms2": ("Cinemática & Operação", "#1E40AF"),
    "mean_neg_accel_ms2": ("Cinemática & Operação", "#1E40AF"),
    "elevation_gain_m": ("Topografia (SRTM)", "#047857"),
    "elevation_loss_m": ("Topografia (SRTM)", "#047857"),
    "net_grade_percent": ("Topografia (SRTM)", "#047857"),
    "aux_hvac_energy_kwh": ("Clima & Conforto (HVAC)", "#D97706"),
    "mean_oat_c": ("Clima & Conforto (HVAC)", "#D97706"),
    "initial_soc_percent": ("Bateria & Eletroquímica", "#7C3AED")
}

TARGET_CONFIGS = {
    "energy_net_kwh": {
        "target_col": "energy_net_kwh",
        "label": "Energia Líquida da Bateria",
        "unit": "kWh",
        "short_name": "Energia [kWh]",
        "envelope_pct": 0.10,
        "model_file": "xgboost_energy_net_model.joblib",
        "is_energy": True
    },
    "delta_soc_percent": {
        "target_col": "delta_soc_percent",
        "label": "Variação do Estado de Carga (ΔSOC)",
        "unit": "%",
        "short_name": "ΔSOC [%]",
        "envelope_pct": 0.15,
        "model_file": "xgboost_delta_soc_model.joblib",
        "is_energy": False
    }
}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, distance_km: np.ndarray, is_energy: bool = True) -> dict:
    """Calcula métricas padronizadas de regressão com normalização por 100km."""
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Trata valores muito próximos de zero no MAPE
    threshold = 0.05 if is_energy else 0.5
    mask_nonzero = y_true >= threshold
    if np.sum(mask_nonzero) > 0:
        mape = float(np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100.0)
    else:
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)

    # Métrica normalizada por 100km
    norm_real = (y_true / np.maximum(distance_km, 0.1)) * 100.0
    norm_pred = (y_pred / np.maximum(distance_km, 0.1)) * 100.0
    mae_100km = float(mean_absolute_error(norm_real, norm_pred))

    res = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_percent": round(mape, 2),
        "mae_100km": round(mae_100km, 2)
    }
    return res


def train_and_evaluate_single_target(
    target_key: str,
    df_valid: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> dict:
    """Treina e avalia o pipeline completo para um alvo específico."""
    cfg = TARGET_CONFIGS[target_key]
    target_col = cfg["target_col"]
    is_energy = cfg["is_energy"]
    unit = cfg["unit"]

    logger.info(f"--- Iniciando Treinamento e Avaliação: {cfg['label']} ({unit}) ---")

    X = df_valid[FEATURE_COLS].copy()
    y = df_valid[target_col].to_numpy(dtype=np.float64)
    distances = df_valid["distance_km"].to_numpy(dtype=np.float64)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    dist_train, dist_test = distances[train_idx], distances[test_idx]

    # 1. 5-Fold Stratified Cross-Validation no Treino
    cv_strat = pd.qcut(dist_train, q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_folds = []
    logger.info("Executando 5-Fold Stratified Cross-Validation no Treino...")

    for fold_num, (f_train_idx, f_val_idx) in enumerate(skf.split(X_train, cv_strat), 1):
        f_X_train, f_X_val = X_train.iloc[f_train_idx], X_train.iloc[f_val_idx]
        f_y_train, f_y_val = y_train[f_train_idx], y_train[f_val_idx]
        f_dist_val = dist_train[f_val_idx]

        model_cv = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        model_cv.fit(f_X_train, f_y_train)
        f_pred = model_cv.predict(f_X_val)

        fold_m = calculate_metrics(f_y_val, f_pred, f_dist_val, is_energy=is_energy)
        cv_folds.append(fold_m)
        logger.info(f"  Fold {fold_num}: R²={fold_m['r2']:.4f} | MAE={fold_m['mae']:.4f} {unit} | RMSE={fold_m['rmse']:.4f} {unit}")

    cv_summary = {
        "mean_r2": float(round(np.mean([f["r2"] for f in cv_folds]), 4)),
        "std_r2": float(round(np.std([f["r2"] for f in cv_folds]), 4)),
        "mean_mae": float(round(np.mean([f["mae"] for f in cv_folds]), 4)),
        "std_mae": float(round(np.std([f["mae"] for f in cv_folds]), 4)),
        "mean_rmse": float(round(np.mean([f["rmse"] for f in cv_folds]), 4)),
        "std_rmse": float(round(np.std([f["rmse"] for f in cv_folds]), 4)),
        "mean_mape_percent": float(round(np.mean([f["mape_percent"] for f in cv_folds]), 2)),
        "mean_mae_100km": float(round(np.mean([f["mae_100km"] for f in cv_folds]), 2)),
        "folds": cv_folds
    }
    logger.info(f"Resumo 5-Fold CV: R² = {cv_summary['mean_r2']} ± {cv_summary['std_r2']}, MAE = {cv_summary['mean_mae']} ± {cv_summary['std_mae']} {unit}")

    # 2. Treinamento do Modelo Final no Treino Completo (N=388)
    logger.info("Treinando Modelo Final XGBoost no conjunto de treino completo...")
    final_model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    # 3. Avaliação no Teste Holdout (N=98)
    y_pred_test = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test, dist_test, is_energy=is_energy)
    logger.info(f"Desempenho no Teste Holdout (20%): R² = {test_metrics['r2']:.4f} | MAE = {test_metrics['mae']:.4f} {unit} | RMSE = {test_metrics['rmse']:.4f} {unit}")

    # 4. Baselines de Comparação
    # Baseline 1: Dummy Mean
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_metrics = calculate_metrics(y_test, dummy_pred, dist_test, is_energy=is_energy)

    # Baseline 2: Regressão Linear Múltipla com as 12 Features
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = calculate_metrics(y_test, lr_pred, dist_test, is_energy=is_energy)

    baselines = {
        "dummy_mean": dummy_metrics,
        "linear_regression_12f": lr_metrics,
        "xgboost_regressor": test_metrics
    }

    # 5. Importância de Variáveis via SHAP
    logger.info("Calculando atribuição de importância SHAP via TreeExplainer...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    shap_ranking = []
    for col, val in zip(X_test.columns, mean_abs_shap):
        group_name, group_color = FEATURE_GROUPS.get(col, ("Outros", "#64748B"))
        shap_ranking.append({
            "feature": col,
            "mean_abs_shap": round(float(val), 5),
            "group": group_name,
            "color": group_color
        })
    shap_ranking.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    # 6. Salva Modelo em Disco
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_MODELS_DIR / cfg["model_file"]
    joblib.dump({
        "model": final_model,
        "features": FEATURE_COLS,
        "target": target_col,
        "target_label": cfg["label"],
        "unit": unit,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary
    }, model_path)
    logger.info(f"Modelo salvo em: {model_path}")

    return {
        "target_key": target_key,
        "target_label": cfg["label"],
        "unit": unit,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "residuals": y_test - y_pred_test,
        "dist_test": dist_test,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "baselines": baselines,
        "shap_ranking": shap_ranking,
        "shap_values": shap_values,
        "X_test": X_test
    }


def plot_dual_real_vs_pred(res_energy: dict, res_soc: dict, save_path: Path):
    """Gera a Figura 25: Painel Dual Real vs. Predito (Energia Líquida e Variação de SOC)."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)

    fig.suptitle(
        "Figura 25: Desempenho Preditivo do XGBoost Regressor Dual no Conjunto de Teste (Holdout 20%)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
        color="#0F172A"
    )

    panels_data = [
        (axes[0], res_energy, "A. Energia Líquida da Bateria ($E_{net}$)", "Consumo Real de Energia [kWh]", "Consumo Predito pelo XGBoost [kWh]", "#1E40AF", "kWh", 0.10),
        (axes[1], res_soc, "B. Variação do Estado de Carga ($\Delta$SOC)", "Variação Real de SOC [$\Delta$%]", "Variação Predita pelo XGBoost [$\Delta$%]", "#7C3AED", "%", 0.15)
    ]

    for ax, res, title, xlabel, ylabel, main_color, unit, env_ratio in panels_data:
        y_real = res["y_test"]
        y_pred = res["y_pred_test"]
        dist = res["dist_test"]
        m = res["test_metrics"]

        max_val = max(np.max(y_real), np.max(y_pred)) * 1.08
        min_val = 0.0

        # Dispersão colorida pela distância da viagem
        scatter = ax.scatter(
            y_real,
            y_pred,
            c=dist,
            cmap="viridis",
            s=48,
            alpha=0.82,
            edgecolors="#0F172A",
            linewidths=0.6,
            zorder=4
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label("Distância da Viagem [km]", fontsize=9, fontweight="semibold")

        # Reta Identidade y = x
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.6, label="Identidade Exata ($y = x$)", zorder=3)

        # Envelope de Erro
        x_env = np.linspace(0.05, max_val, 100)
        ax.fill_between(
            x_env,
            x_env * (1.0 - env_ratio),
            x_env * (1.0 + env_ratio),
            color="#94A3B8",
            alpha=0.18,
            label=f"Envelope $\pm${int(env_ratio*100)}%"
        )

        # Caixa com Métricas Oficiais
        norm_text = f"• MAE/100km: {m['mae_100km']:.2f} {unit}/100km\n" if unit == "kWh" else ""
        box_text = (
            f"Métricas Holdout (N=98 viagens):\n"
            f"• R² = {m['r2']:.4f}\n"
            f"• MAE = {m['mae']:.3f} {unit}\n"
            f"• RMSE = {m['rmse']:.3f} {unit}\n"
            f"• MAPE = {m['mape_percent']:.1f}%\n"
            f"{norm_text}"
            f"• 5-Fold CV R²: {res['cv_summary']['mean_r2']:.4f} ± {res['cv_summary']['std_r2']:.3f}"
        )
        ax.text(
            0.04, 0.96, box_text,
            transform=ax.transAxes,
            fontsize=9.2,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor=main_color, alpha=0.92, linewidth=1.2),
            zorder=5
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8, color="#0F172A")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold")
        ax.legend(loc="lower right", fontsize=8.8, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura 25 salva com sucesso: {save_path}")


def plot_dual_residuals_distribution(res_energy: dict, res_soc: dict, save_path: Path):
    """Gera a Figura 26: Painel Dual da Distribuição dos Resíduos de Predição."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)

    fig.suptitle(
        "Figura 26: Distribuição e Densidade dos Resíduos de Predição ($e = y_{real} - y_{pred}$)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
        color="#0F172A"
    )

    panels_data = [
        (axes[0], res_energy, "A. Resíduos da Energia Líquida ($E_{net}$)", "Erro Residual de Energia [kWh]", "#1E40AF", "kWh"),
        (axes[1], res_soc, "B. Resíduos da Variação de SOC ($\Delta$SOC)", "Erro Residual de SOC [$\Delta$%]", "#7C3AED", "%")
    ]

    for ax, res, title, xlabel, color, unit in panels_data:
        residuals = res["residuals"]
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        median_res = float(np.median(residuals))
        p05 = float(np.percentile(residuals, 5))
        p95 = float(np.percentile(residuals, 95))

        sns.histplot(
            residuals,
            kde=True,
            bins=22,
            color=color,
            edgecolor="#0F172A",
            alpha=0.6,
            ax=ax,
            linewidth=0.8
        )

        ax.axvline(0.0, color="black", linestyle="--", lw=1.6, label="Erro Nulo (0.0)")
        ax.axvline(mean_res, color="#DC2626", linestyle="-", lw=1.6, label=f"Média: {mean_res:+.3f} {unit}")
        ax.axvline(median_res, color="#D97706", linestyle=":", lw=1.6, label=f"Mediana: {median_res:+.3f} {unit}")
        ax.axvline(p05, color="#64748B", linestyle="-.", lw=1.1, label=f"P5-P95: [{p05:+.2f}, {p95:+.2f}]")
        ax.axvline(p95, color="#64748B", linestyle="-.", lw=1.1)

        stats_box = (
            f"Diagnóstico Residual (N=98):\n"
            f"• Média:    {mean_res:+.4f} {unit}\n"
            f"• Desv Padrão: {std_res:.4f} {unit}\n"
            f"• Mediana:  {median_res:+.4f} {unit}\n"
            f"• Int. 90%: [{p05:+.3f}, {p95:+.3f}]"
        )
        ax.text(
            0.04, 0.94, stats_box,
            transform=ax.transAxes,
            fontsize=9.2,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor=color, alpha=0.92, linewidth=1.2)
        )

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8, color="#0F172A")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold")
        ax.set_ylabel("Frequência de Viagens de Teste", fontsize=10, fontweight="semibold")
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura 26 salva com sucesso: {save_path}")


def plot_dual_shap_importance(res_energy: dict, res_soc: dict, save_path: Path):
    """Gera a Figura 27: Painel Dual de Importância SHAP das 12 Features Físicas."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

    fig.suptitle(
        "Figura 27: Atribuição de Importância das 12 Features Físicas (SHAP TreeExplainer Attribution)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
        color="#0F172A"
    )

    panels_data = [
        (axes[0], res_energy, "A. Importância para Energia Líquida ($E_{net}$)", "Impacto Médio Absoluto [mean(|SHAP|) em kWh]", "kWh"),
        (axes[1], res_soc, "B. Importância para Variação de SOC ($\Delta$SOC)", "Impacto Médio Absoluto [mean(|SHAP|) em %]", "%")
    ]

    for ax, res, title, xlabel, unit in panels_data:
        ranking = res["shap_ranking"]
        df_plot = pd.DataFrame(ranking).sort_values("mean_abs_shap", ascending=True)

        y_pos = np.arange(len(df_plot))
        colors = df_plot["color"].tolist()

        bars = ax.barh(y_pos, df_plot["mean_abs_shap"], color=colors, height=0.72, edgecolor="#0F172A", linewidth=0.6)

        # Adiciona rótulos numéricos
        for i, val in enumerate(df_plot["mean_abs_shap"]):
            ax.text(val + 0.002 * max(df_plot["mean_abs_shap"]), i, f"{val:.4f} {unit}", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot["feature"], fontsize=9.5, fontweight="semibold")
        ax.set_xlabel(xlabel, fontsize=9.8, fontweight="semibold")
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=8, color="#0F172A")
        ax.set_xlim(0, max(df_plot["mean_abs_shap"]) * 1.25)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")

    # Legenda Compartilhada de Grupos Físicos
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#1E40AF", label="Cinemática & Operação"),
        plt.Rectangle((0, 0), 1, 1, color="#047857", label="Topografia (SRTM)"),
        plt.Rectangle((0, 0), 1, 1, color="#D97706", label="Clima & Conforto (HVAC)"),
        plt.Rectangle((0, 0), 1, 1, color="#7C3AED", label="Bateria & Eletroquímica")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, framealpha=0.92, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura 27 salva com sucesso: {save_path}")


def run_dual_xgboost_pipeline():
    """Executa o pipeline completo para os dois alvos."""
    logger.info("=== Carregando Base Consolidada de Viagens Completas do VED ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo {DATA_PATH} não encontrado.")

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Total de viagens registradas: {len(df)}")

    # Filtro de viagens físicas válidas
    df_valid = df[df["is_valid_trip"]].copy().reset_index(drop=True)
    n_valid = len(df_valid)
    logger.info(f"Viagens válidas para modelagem (distância >= 0.5 km, duração >= 60s): {n_valid}")

    # Divisão Estratificada 80% Treino / 20% Teste Holdout baseada em quantis de distância
    dist_bins = pd.qcut(df_valid["distance_km"], q=5, labels=False)
    train_idx, test_idx = train_test_split(
        np.arange(n_valid),
        test_size=0.20,
        random_state=42,
        stratify=dist_bins
    )
    logger.info(f"Divisão concluída: Treino = {len(train_idx)} viagens (80%) | Teste Holdout = {len(test_idx)} viagens (20%)")

    # Executa Modelagem Dual
    res_energy = train_and_evaluate_single_target("energy_net_kwh", df_valid, train_idx, test_idx)
    res_soc = train_and_evaluate_single_target("delta_soc_percent", df_valid, train_idx, test_idx)

    # Gera Figuras Mestres Dual (300 DPI)
    OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig25_path = OUTPUT_FIGS_DIR / "25_xgboost_real_vs_pred_energy.png"
    fig26_path = OUTPUT_FIGS_DIR / "26_xgboost_residuals_distribution.png"
    fig27_path = OUTPUT_FIGS_DIR / "27_xgboost_feature_importance_shap.png"

    plot_dual_real_vs_pred(res_energy, res_soc, fig25_path)
    plot_dual_residuals_distribution(res_energy, res_soc, fig26_path)
    plot_dual_shap_importance(res_energy, res_soc, fig27_path)

    # Copia figuras para diretório de artefatos
    if ARTIFACT_DIR.exists():
        for src_fig in [fig25_path, fig26_path, fig27_path]:
            dst_fig = ARTIFACT_DIR / src_fig.name
            shutil.copy2(src_fig, dst_fig)
            logger.info(f"Artefato copiado para brain: {dst_fig}")

    # Salva Relatórios Consolidados em JSON e CSV
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = {
        "dataset_metadata": {
            "total_trips": len(df),
            "valid_trips": n_valid,
            "train_trips": len(train_idx),
            "test_trips": len(test_idx),
            "n_features": len(FEATURE_COLS),
            "feature_columns": FEATURE_COLS
        },
        "models": {
            "energy_net_kwh": {
                "label": res_energy["target_label"],
                "unit": res_energy["unit"],
                "cv_5fold": res_energy["cv_summary"],
                "holdout_test": res_energy["test_metrics"],
                "baselines": res_energy["baselines"],
                "shap_ranking": res_energy["shap_ranking"]
            },
            "delta_soc_percent": {
                "label": res_soc["target_label"],
                "unit": res_soc["unit"],
                "cv_5fold": res_soc["cv_summary"],
                "holdout_test": res_soc["test_metrics"],
                "baselines": res_soc["baselines"],
                "shap_ranking": res_soc["shap_ranking"]
            }
        }
    }

    json_path = OUTPUT_RESULTS_DIR / "xgboost_dual_model_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório consolidado JSON salvo: {json_path}")

    # Tabela Comparativa em CSV
    rows_comp = []
    for model_key, r in [("energy_net_kwh", res_energy), ("delta_soc_percent", res_soc)]:
        for m_name, metrics in r["baselines"].items():
            rows_comp.append({
                "target": model_key,
                "target_label": r["target_label"],
                "unit": r["unit"],
                "model_name": m_name,
                "r2_test": metrics["r2"],
                "mae_test": metrics["mae"],
                "rmse_test": metrics["rmse"],
                "mape_percent_test": metrics["mape_percent"],
                "mae_100km_test": metrics["mae_100km"]
            })
    df_comp = pd.DataFrame(rows_comp)
    csv_path = OUTPUT_RESULTS_DIR / "xgboost_dual_model_evaluation.csv"
    df_comp.to_csv(csv_path, index=False)
    logger.info(f"Tabela comparativa CSV salva: {csv_path}")

    # Salva Ranking SHAP Consolidado em CSV
    shap_rows = []
    for model_key, r in [("energy_net_kwh", res_energy), ("delta_soc_percent", res_soc)]:
        for item in r["shap_ranking"]:
            shap_rows.append({
                "target": model_key,
                "feature": item["feature"],
                "group": item["group"],
                "mean_abs_shap": item["mean_abs_shap"],
                "unit": r["unit"]
            })
    df_shap = pd.DataFrame(shap_rows)
    shap_csv_path = OUTPUT_RESULTS_DIR / "xgboost_shap_rankings.csv"
    df_shap.to_csv(shap_csv_path, index=False)
    logger.info(f"Ranking SHAP CSV salvo: {shap_csv_path}")

    print("\n" + "=" * 75)
    print("RESUMO EXECUTIVO: MODELAGEM DUAL DE CONSUMO COM XGBOOST REGRESSOR")
    print("=" * 75)
    print("\n1. MODELO DE ENERGIA LÍQUIDA (Target = energy_net_kwh [kWh]):")
    print(f"   5-Fold CV Treino:   R² = {res_energy['cv_summary']['mean_r2']:.4f} ± {res_energy['cv_summary']['std_r2']:.4f} | MAE = {res_energy['cv_summary']['mean_mae']:.4f} kWh")
    print(f"   Holdout Teste 20%: R² = {res_energy['test_metrics']['r2']:.4f} | MAE = {res_energy['test_metrics']['mae']:.4f} kWh ({res_energy['test_metrics']['mae']*1000:.1f} Wh)")
    print(f"   RMSE Teste:        {res_energy['test_metrics']['rmse']:.4f} kWh | MAPE = {res_energy['test_metrics']['mape_percent']:.1f}%")
    print(f"   MAE Normalizado:   {res_energy['test_metrics']['mae_100km']:.2f} kWh/100km")
    print("   Top 3 SHAP Features:")
    for item in res_energy["shap_ranking"][:3]:
        print(f"     - {item['feature']:20s} [{item['group']}]: {item['mean_abs_shap']:.4f} kWh")

    print("\n2. MODELO DE ESTADO DE CARGA (Target = delta_soc_percent [%]):")
    print(f"   5-Fold CV Treino:   R² = {res_soc['cv_summary']['mean_r2']:.4f} ± {res_soc['cv_summary']['std_r2']:.4f} | MAE = {res_soc['cv_summary']['mean_mae']:.4f} %")
    print(f"   Holdout Teste 20%: R² = {res_soc['test_metrics']['r2']:.4f} | MAE = {res_soc['test_metrics']['mae']:.4f} %")
    print(f"   RMSE Teste:        {res_soc['test_metrics']['rmse']:.4f} % | MAPE = {res_soc['test_metrics']['mape_percent']:.1f}%")
    print("   Top 3 SHAP Features:")
    for item in res_soc["shap_ranking"][:3]:
        print(f"     - {item['feature']:20s} [{item['group']}]: {item['mean_abs_shap']:.4f} %")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_dual_xgboost_pipeline()
