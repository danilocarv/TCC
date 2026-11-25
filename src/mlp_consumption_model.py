"""
Módulo da Etapa 1 (Diagnóstico Base) — MLP Regressor para Consumo Energético.

Implementa o modelo base de Rede Neural Artificial (MLPRegressor) encapsulado em
Pipeline com StandardScaler para ambos os alvos (Energia Líquida e ΔSOC), utilizando
exatamente a mesma divisão estratificada (388 treino / 98 holdout, seed=42) e os
mesmos 5 folds de validação cruzada da Fase 2.

Na Etapa 1, avaliamos convergência, estabilidade na CV 5-fold, diferença entre
desempenho no treino e na CV (diagnóstico de overfitting) e eventuais previsões
negativas fora da amostra de treino (out-of-fold), mantendo o holdout reservado
para a Etapa 3.
"""

import argparse
import itertools
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xgboost_consumption_model import (  # noqa: E402
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURE_COLS,
    OUTPUT_FIGS_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_RESULTS_DIR,
    TARGET_CONFIGS,
    _build_shap_ranking,
    _fmt_pt,
    _run_kfold_cv,
    calculate_metrics,
    plot_dual_real_vs_pred,
    plot_dual_residuals_distribution,
    plot_dual_shap_importance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuração base enxuta da Etapa 1
MLP_BASE_PARAMS = {
    "hidden_layer_sizes": (16,),
    "activation": "relu",
    "solver": "lbfgs",
    "alpha": 0.1,
    "max_iter": 2000,
    "random_state": 42,
}


def make_mlp_pipeline(params: dict | None = None, scale_target: bool = False):
    """Cria Pipeline com padronização Z-score em X (e opcionalmente em y via TransformedTargetRegressor)."""
    cfg = dict(MLP_BASE_PARAMS)
    if params:
        cfg.update(params)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(**cfg)),
    ])
    if scale_target:
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return pipe


def _collect_oof_predictions(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    dist_train: np.ndarray,
    params: dict | None = None,
) -> tuple[np.ndarray, int]:
    """Gera predições out-of-fold na CV 5-fold e conta avisos de convergência."""
    cv_strat = pd.qcut(dist_train, q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y_train, dtype=np.float64)
    conv_warnings = 0

    for f_train_idx, f_val_idx in skf.split(X_train, cv_strat):
        pipe = make_mlp_pipeline(params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(X_train.iloc[f_train_idx], y_train[f_train_idx])
            conv_warnings += sum(
                1 for w in caught if issubclass(w.category, ConvergenceWarning)
            )
        oof_preds[f_val_idx] = pipe.predict(X_train.iloc[f_val_idx])

    return oof_preds, conv_warnings


def run_mlp_stage1_diagnostic() -> dict:
    """Executa a Etapa 1 do MLP (treino vs. CV 5-fold + diagnóstico de convergência)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo {DATA_PATH} não encontrado.")

    df = pd.read_parquet(DATA_PATH)
    df_valid = df[df["is_valid_trip"]].copy().reset_index(drop=True)

    dist_bins = pd.qcut(df_valid["distance_km"], q=5, labels=False)
    train_idx, test_idx = train_test_split(
        np.arange(len(df_valid)),
        test_size=0.20,
        random_state=42,
        stratify=dist_bins,
    )

    X_train = df_valid[FEATURE_COLS].iloc[train_idx].copy()
    dist_train = df_valid["distance_km"].to_numpy(dtype=np.float64)[train_idx]

    report = {
        "stage": "Etapa 1 - Implementação base e diagnóstico do MLP",
        "model_params": {
            "hidden_layer_sizes": list(MLP_BASE_PARAMS["hidden_layer_sizes"]),
            "activation": MLP_BASE_PARAMS["activation"],
            "solver": MLP_BASE_PARAMS["solver"],
            "alpha": MLP_BASE_PARAMS["alpha"],
            "max_iter": MLP_BASE_PARAMS["max_iter"],
            "random_state": MLP_BASE_PARAMS["random_state"],
        },
        "dataset_metadata": {
            "valid_trips": int(len(df_valid)),
            "train_trips": int(len(train_idx)),
            "holdout_trips_reserved": int(len(test_idx)),
            "n_features": int(len(FEATURE_COLS)),
        },
        "targets": {},
    }

    for target_key in ("energy_net_kwh", "delta_soc_percent"):
        cfg = TARGET_CONFIGS[target_key]
        y_train = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)[train_idx]
        is_energy = cfg["is_energy"]
        unit = cfg["unit"]

        logger.info(f"=== Etapa 1 MLP: {cfg['label']} ({unit}) ===")
        cv_summary = _run_kfold_cv(
            make_mlp_pipeline, X_train, y_train, dist_train, is_energy, unit, "MLP (Etapa 1)"
        )
        oof_preds, oof_conv_warnings = _collect_oof_predictions(X_train, y_train, dist_train)

        # Ajuste no conjunto completo de treino (388 viagens) para comparar Treino vs. CV
        full_pipe = make_mlp_pipeline()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            full_pipe.fit(X_train, y_train)
            full_conv_warnings = sum(
                1 for w in caught if issubclass(w.category, ConvergenceWarning)
            )

        train_preds = full_pipe.predict(X_train)
        train_metrics = calculate_metrics(y_train, train_preds, dist_train, is_energy=is_energy)
        mlp_step = full_pipe.named_steps["mlp"]

        report["targets"][target_key] = {
            "label": cfg["label"],
            "unit": unit,
            "train_in_sample": train_metrics,
            "cv_5fold": cv_summary,
            "overfitting_gap": {
                "r2_train_minus_cv": round(train_metrics["r2"] - cv_summary["mean_r2"], 4),
                "mae_cv_minus_train": round(cv_summary["mean_mae"] - train_metrics["mae"], 4),
            },
            "convergence": {
                "n_iter_full_train": int(mlp_step.n_iter_),
                "loss_full_train": round(float(mlp_step.loss_), 6),
                "warnings_cv_folds": int(oof_conv_warnings),
                "warnings_full_train": int(full_conv_warnings),
            },
            "oof_negative_predictions": {
                "count": int(np.sum(oof_preds < 0.0)),
                "min_pred": round(float(np.min(oof_preds)), 4),
            },
        }

    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_RESULTS_DIR / "mlp_etapa1_diagnostic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório da Etapa 1 salvo em: {out_path}")

    return report


# Grade controlada da Etapa 2 (avaliada exclusivamente na CV 5-fold do treino)
MLP_STAGE2_GRID = {
    "hidden_layer_sizes": [(8,), (16,), (16, 8)],
    "activation": ["relu", "tanh"],
    "alpha": [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
}


def _load_train_split() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, int]:
    """Carrega base válida e retorna apenas a partição de treino (80%), mantendo holdout reservado."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo {DATA_PATH} não encontrado.")

    df = pd.read_parquet(DATA_PATH)
    df_valid = df[df["is_valid_trip"]].copy().reset_index(drop=True)

    dist_bins = pd.qcut(df_valid["distance_km"], q=5, labels=False)
    train_idx, test_idx = train_test_split(
        np.arange(len(df_valid)),
        test_size=0.20,
        random_state=42,
        stratify=dist_bins,
    )
    return df_valid, train_idx, test_idx, len(df_valid), len(test_idx)


def _evaluate_candidate_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    dist_train: np.ndarray,
    is_energy: bool,
    params: dict,
    scale_target: bool = False,
) -> dict:
    """Avalia uma configuração de MLP na 5-Fold Stratified CV + treino completo (diagnóstico)."""
    cv_strat = pd.qcut(dist_train, q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    oof_preds = np.zeros_like(y_train, dtype=np.float64)
    cv_warnings = 0

    for f_train_idx, f_val_idx in skf.split(X_train, cv_strat):
        model = make_mlp_pipeline(params, scale_target=scale_target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train.iloc[f_train_idx], y_train[f_train_idx])
            cv_warnings += sum(
                1 for w in caught if issubclass(w.category, ConvergenceWarning)
            )
        val_pred = model.predict(X_train.iloc[f_val_idx])
        oof_preds[f_val_idx] = val_pred
        fold_metrics.append(
            calculate_metrics(y_train[f_val_idx], val_pred, dist_train[f_val_idx], is_energy=is_energy)
        )

    full_model = make_mlp_pipeline(params, scale_target=scale_target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        full_model.fit(X_train, y_train)
        full_warnings = sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning)
        )
    train_preds = full_model.predict(X_train)
    train_m = calculate_metrics(y_train, train_preds, dist_train, is_energy=is_energy)
    mlp_step = (
        full_model.regressor_.named_steps["mlp"]
        if scale_target
        else full_model.named_steps["mlp"]
    )

    mean_r2 = float(round(np.mean([f["r2"] for f in fold_metrics]), 4))
    std_r2 = float(round(np.std([f["r2"] for f in fold_metrics]), 4))
    mean_mae = float(round(np.mean([f["mae"] for f in fold_metrics]), 4))
    std_mae = float(round(np.std([f["mae"] for f in fold_metrics]), 4))
    mean_rmse = float(round(np.mean([f["rmse"] for f in fold_metrics]), 4))
    std_rmse = float(round(np.std([f["rmse"] for f in fold_metrics]), 4))
    mean_mape = float(round(np.mean([f["mape_percent"] for f in fold_metrics]), 2))
    mean_mae_100km = float(round(np.mean([f["mae_100km"] for f in fold_metrics]), 2))

    return {
        "params": {
            "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
            "activation": params["activation"],
            "alpha": params["alpha"],
            "solver": params["solver"],
            "max_iter": params["max_iter"],
            "random_state": params["random_state"],
        },
        "cv_r2_mean": mean_r2,
        "cv_r2_std": std_r2,
        "cv_mae_mean": mean_mae,
        "cv_mae_std": std_mae,
        "cv_rmse_mean": mean_rmse,
        "cv_rmse_std": std_rmse,
        "cv_mape_mean": mean_mape,
        "cv_mae_100km_mean": mean_mae_100km,
        "train_r2": train_m["r2"],
        "train_mae": train_m["mae"],
        "r2_gap_train_minus_cv": round(train_m["r2"] - mean_r2, 4),
        "n_iter_full_train": int(mlp_step.n_iter_),
        "warnings_cv_folds": int(cv_warnings),
        "warnings_full_train": int(full_warnings),
        "oof_neg_count": int(np.sum(oof_preds < 0.0)),
        "oof_min_pred": round(float(np.min(oof_preds)), 4),
    }


def run_mlp_stage2_cv_tuning() -> dict:
    """Executa a Etapa 2: busca controlada na CV 5-fold do treino (sem tocar no holdout)."""
    df_valid, train_idx, test_idx, n_valid, n_holdout = _load_train_split()
    X_train = df_valid[FEATURE_COLS].iloc[train_idx].copy()
    dist_train = df_valid["distance_km"].to_numpy(dtype=np.float64)[train_idx]

    combos = list(itertools.product(
        MLP_STAGE2_GRID["hidden_layer_sizes"],
        MLP_STAGE2_GRID["activation"],
        MLP_STAGE2_GRID["alpha"],
    ))
    logger.info(
        f"Iniciando Etapa 2 MLP: {len(combos)} combinações × 5 folds por alvo "
        f"(treino={len(train_idx)}, holdout={n_holdout} reservado)."
    )

    report = {
        "stage": "Etapa 2 - Ajuste controlado na CV 5-fold do treino",
        "grid": {
            "hidden_layer_sizes": [list(h) for h in MLP_STAGE2_GRID["hidden_layer_sizes"]],
            "activation": MLP_STAGE2_GRID["activation"],
            "alpha": MLP_STAGE2_GRID["alpha"],
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": 42,
        },
        "dataset_metadata": {
            "valid_trips": int(n_valid),
            "train_trips": int(len(train_idx)),
            "holdout_trips_reserved": int(n_holdout),
            "n_features": int(len(FEATURE_COLS)),
        },
        "targets": {},
    }

    csv_rows = []
    for target_key in ("energy_net_kwh", "delta_soc_percent"):
        cfg = TARGET_CONFIGS[target_key]
        y_train = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)[train_idx]
        is_energy = cfg["is_energy"]
        unit = cfg["unit"]

        logger.info(f"=== Etapa 2 CV Tuning: {cfg['label']} ({unit}) ===")
        candidates = []
        for hidden, act, alpha in combos:
            params = {
                "hidden_layer_sizes": hidden,
                "activation": act,
                "solver": "lbfgs",
                "alpha": alpha,
                "max_iter": 5000,
                "random_state": 42,
            }
            res = _evaluate_candidate_cv(X_train, y_train, dist_train, is_energy, params)
            candidates.append(res)
            csv_rows.append({
                "target": target_key,
                "hidden_layer_sizes": str(hidden),
                "activation": act,
                "alpha": alpha,
                "cv_r2_mean": res["cv_r2_mean"],
                "cv_r2_std": res["cv_r2_std"],
                "cv_mae_mean": res["cv_mae_mean"],
                "cv_rmse_mean": res["cv_rmse_mean"],
                "cv_mape_mean": res["cv_mape_mean"],
                "train_r2": res["train_r2"],
                "r2_gap_train_minus_cv": res["r2_gap_train_minus_cv"],
                "warnings_cv_folds": res["warnings_cv_folds"],
                "warnings_full_train": res["warnings_full_train"],
                "n_iter_full_train": res["n_iter_full_train"],
                "oof_neg_count": res["oof_neg_count"],
                "oof_min_pred": res["oof_min_pred"],
            })

        # Ordena priorizando convergência limpa, maior R² médio na CV e menor MAE médio
        candidates.sort(
            key=lambda x: (
                x["warnings_cv_folds"] > 0 or x["warnings_full_train"] > 0,
                -x["cv_r2_mean"],
                x["cv_mae_mean"],
            )
        )
        best = candidates[0]
        logger.info(
            f"Melhor [{target_key}]: hidden={tuple(best['params']['hidden_layer_sizes'])}, "
            f"act={best['params']['activation']}, alpha={best['params']['alpha']} -> "
            f"CV R²={best['cv_r2_mean']:.4f} ± {best['cv_r2_std']:.4f}, "
            f"CV MAE={best['cv_mae_mean']:.4f} {unit}, "
            f"gap={best['r2_gap_train_minus_cv']:+.4f}, "
            f"avisos={best['warnings_cv_folds']}/{best['warnings_full_train']}, "
            f"neg_OOF={best['oof_neg_count']}"
        )

        report["targets"][target_key] = {
            "label": cfg["label"],
            "unit": unit,
            "best_candidate": best,
            "top_5_candidates": candidates[:5],
            "all_candidates_count": len(candidates),
        }

    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_RESULTS_DIR / "mlp_etapa2_cv_tuning.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório da Etapa 2 salvo em: {json_path}")

    csv_path = OUTPUT_RESULTS_DIR / "mlp_etapa2_cv_tuning.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Tabela completa da Etapa 2 salva em: {csv_path}")

    return report


def run_mlp_stage2b_refinement() -> dict:
    """Etapa 2b: compara Opção 1 (alpha estendido no ΔSOC) e Opção 2 (TransformedTargetRegressor)."""
    df_valid, train_idx, _test_idx, n_valid, n_holdout = _load_train_split()
    X_train = df_valid[FEATURE_COLS].iloc[train_idx].copy()
    dist_train = df_valid["distance_km"].to_numpy(dtype=np.float64)[train_idx]

    report = {
        "stage": "Etapa 2b - Refinamento na CV 5-fold (Opção 1: alpha estendido ΔSOC | Opção 2: alvo y padronizado)",
        "dataset_metadata": {
            "valid_trips": int(n_valid),
            "train_trips": int(len(train_idx)),
            "holdout_trips_reserved": int(n_holdout),
            "n_features": int(len(FEATURE_COLS)),
        },
        "option1_delta_soc_extended_alpha": [],
        "option2_scaled_target": {},
    }
    csv_rows = []

    def _sort_key(item: dict):
        return (
            item["warnings_cv_folds"] > 0 or item["warnings_full_train"] > 0,
            -item["cv_r2_mean"],
            item["cv_mae_mean"],
        )

    # Opção 1: Estender alpha no ΔSOC para (16, 8) sem escalar y
    logger.info("=== Opção 1: Extensão de alpha para (16, 8) no ΔSOC (sem escalar y) ===")
    y_soc = df_valid["delta_soc_percent"].to_numpy(dtype=np.float64)[train_idx]
    opt1_alphas = [20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
    for act in ("tanh", "relu"):
        for alpha in opt1_alphas:
            params = {
                "hidden_layer_sizes": (16, 8),
                "activation": act,
                "solver": "lbfgs",
                "alpha": alpha,
                "max_iter": 5000,
                "random_state": 42,
            }
            res = _evaluate_candidate_cv(X_train, y_soc, dist_train, False, params, scale_target=False)
            report["option1_delta_soc_extended_alpha"].append(res)
            csv_rows.append({
                "experiment": "option1_raw_y_extended",
                "target": "delta_soc_percent",
                "scale_target": False,
                "hidden_layer_sizes": "(16, 8)",
                "activation": act,
                "alpha": alpha,
                "cv_r2_mean": res["cv_r2_mean"],
                "cv_r2_std": res["cv_r2_std"],
                "cv_mae_mean": res["cv_mae_mean"],
                "cv_rmse_mean": res["cv_rmse_mean"],
                "cv_mape_mean": res["cv_mape_mean"],
                "train_r2": res["train_r2"],
                "r2_gap_train_minus_cv": res["r2_gap_train_minus_cv"],
                "warnings_cv_folds": res["warnings_cv_folds"],
                "warnings_full_train": res["warnings_full_train"],
                "n_iter_full_train": res["n_iter_full_train"],
                "oof_neg_count": res["oof_neg_count"],
                "oof_min_pred": res["oof_min_pred"],
            })

    report["option1_delta_soc_extended_alpha"].sort(key=_sort_key)
    best_opt1 = report["option1_delta_soc_extended_alpha"][0]
    logger.info(
        f"Melhor Opção 1 [ΔSOC]: (16, 8), {best_opt1['params']['activation']}, "
        f"alpha={best_opt1['params']['alpha']} -> CV R²={best_opt1['cv_r2_mean']:.4f} ± "
        f"{best_opt1['cv_r2_std']:.4f}, MAE={best_opt1['cv_mae_mean']:.4f} %, "
        f"gap={best_opt1['r2_gap_train_minus_cv']:+.4f}, neg_OOF={best_opt1['oof_neg_count']}"
    )

    # Opção 2: Padronizar também o alvo y (TransformedTargetRegressor) em ambos os alvos
    opt2_combos = list(itertools.product(
        [(8,), (16,), (16, 8)],
        ["relu", "tanh"],
        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    ))
    for target_key in ("energy_net_kwh", "delta_soc_percent"):
        cfg = TARGET_CONFIGS[target_key]
        y_train = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)[train_idx]
        is_energy = cfg["is_energy"]
        unit = cfg["unit"]
        logger.info(f"=== Opção 2 (y padronizado): {cfg['label']} ({len(opt2_combos)} combinações) ===")

        candidates = []
        for hidden, act, alpha in opt2_combos:
            params = {
                "hidden_layer_sizes": hidden,
                "activation": act,
                "solver": "lbfgs",
                "alpha": alpha,
                "max_iter": 5000,
                "random_state": 42,
            }
            res = _evaluate_candidate_cv(X_train, y_train, dist_train, is_energy, params, scale_target=True)
            candidates.append(res)
            csv_rows.append({
                "experiment": "option2_scaled_y",
                "target": target_key,
                "scale_target": True,
                "hidden_layer_sizes": str(hidden),
                "activation": act,
                "alpha": alpha,
                "cv_r2_mean": res["cv_r2_mean"],
                "cv_r2_std": res["cv_r2_std"],
                "cv_mae_mean": res["cv_mae_mean"],
                "cv_rmse_mean": res["cv_rmse_mean"],
                "cv_mape_mean": res["cv_mape_mean"],
                "train_r2": res["train_r2"],
                "r2_gap_train_minus_cv": res["r2_gap_train_minus_cv"],
                "warnings_cv_folds": res["warnings_cv_folds"],
                "warnings_full_train": res["warnings_full_train"],
                "n_iter_full_train": res["n_iter_full_train"],
                "oof_neg_count": res["oof_neg_count"],
                "oof_min_pred": res["oof_min_pred"],
            })

        candidates.sort(key=_sort_key)
        best = candidates[0]
        logger.info(
            f"Melhor Opção 2 [{target_key}]: hidden={tuple(best['params']['hidden_layer_sizes'])}, "
            f"act={best['params']['activation']}, alpha={best['params']['alpha']} -> "
            f"CV R²={best['cv_r2_mean']:.4f} ± {best['cv_r2_std']:.4f}, "
            f"MAE={best['cv_mae_mean']:.4f} {unit}, gap={best['r2_gap_train_minus_cv']:+.4f}, "
            f"avisos={best['warnings_cv_folds']}/{best['warnings_full_train']}, neg_OOF={best['oof_neg_count']}"
        )
        report["option2_scaled_target"][target_key] = {
            "label": cfg["label"],
            "unit": unit,
            "best_candidate": best,
            "top_5_candidates": candidates[:5],
        }

    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_RESULTS_DIR / "mlp_etapa2b_refinement.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório da Etapa 2b salvo em: {json_path}")

    csv_path = OUTPUT_RESULTS_DIR / "mlp_etapa2b_refinement.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Tabela da Etapa 2b salva em: {csv_path}")

    return report


# Configurações finais selecionadas na Etapa 2 / 2b (Opção 1 — sem padronizar y)
MLP_FINAL_CONFIGS = {
    "energy_net_kwh": {
        "hidden_layer_sizes": (16, 8),
        "activation": "relu",
        "solver": "lbfgs",
        "alpha": 1.0,
        "max_iter": 5000,
        "random_state": 42,
        "model_file": "mlp_energy_net_model.joblib",
    },
    "delta_soc_percent": {
        "hidden_layer_sizes": (16, 8),
        "activation": "tanh",
        "solver": "lbfgs",
        "alpha": 20.0,
        "max_iter": 5000,
        "random_state": 42,
        "model_file": "mlp_delta_soc_model.joblib",
    },
}


def run_mlp_stage3_final_evaluation() -> dict:
    """Etapa 3: treino final (388), avaliação no holdout (98), SHAP KernelExplainer e gráficos."""
    df_valid, train_idx, test_idx, n_valid, n_holdout = _load_train_split()
    X = df_valid[FEATURE_COLS].copy()
    distances = df_valid["distance_km"].to_numpy(dtype=np.float64)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    dist_train, dist_test = distances[train_idx], distances[test_idx]

    target_results = {}
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Background resumido com k-means (50 centróides) para SHAP KernelExplainer determinístico
    np.random.seed(42)
    shap_bg = shap.kmeans(X_train, 50)

    for target_key in ("energy_net_kwh", "delta_soc_percent"):
        cfg = TARGET_CONFIGS[target_key]
        mlp_cfg = dict(MLP_FINAL_CONFIGS[target_key])
        model_file = mlp_cfg.pop("model_file")
        is_energy = cfg["is_energy"]
        unit = cfg["unit"]

        y = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(
            f"=== Etapa 3 MLP Final: {cfg['label']} "
            f"(hidden={mlp_cfg['hidden_layer_sizes']}, act={mlp_cfg['activation']}, alpha={mlp_cfg['alpha']}) ==="
        )
        cv_summary = _run_kfold_cv(
            lambda c=mlp_cfg: make_mlp_pipeline(c, scale_target=False),
            X_train,
            y_train,
            dist_train,
            is_energy,
            unit,
            "MLP Final",
        )

        final_model = make_mlp_pipeline(mlp_cfg, scale_target=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            final_model.fit(X_train, y_train)
            conv_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

        train_pred = final_model.predict(X_train)
        train_metrics = calculate_metrics(y_train, train_pred, dist_train, is_energy=is_energy)

        y_pred_test = final_model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred_test, dist_test, is_energy=is_energy)
        residuals = y_test - y_pred_test

        logger.info(
            f"Holdout MLP [{target_key}]: R²={test_metrics['r2']:.4f} | "
            f"MAE={test_metrics['mae']:.4f} {unit} | RMSE={test_metrics['rmse']:.4f} {unit} | "
            f"MAPE={test_metrics['mape_percent']:.2f}% | neg={int(np.sum(y_pred_test < 0))}"
        )

        # SHAP KernelExplainer nas 98 viagens de teste
        logger.info(f"Calculando SHAP KernelExplainer para MLP [{target_key}]...")
        np.random.seed(42)

        def _predict_fn(arr, m=final_model):
            return m.predict(pd.DataFrame(arr, columns=FEATURE_COLS))

        explainer = shap.KernelExplainer(_predict_fn, shap_bg)
        shap_values = explainer.shap_values(X_test, nsamples=200, silent=True)
        shap_ranking = _build_shap_ranking(np.asarray(shap_values), X_test.columns)

        model_path = OUTPUT_MODELS_DIR / model_file
        joblib.dump(
            {
                "model": final_model,
                "params": mlp_cfg,
                "features": FEATURE_COLS,
                "target": cfg["target_col"],
                "target_label": cfg["label"],
                "unit": unit,
                "cv_summary": cv_summary,
                "test_metrics": test_metrics,
            },
            model_path,
        )
        logger.info(f"Modelo MLP salvo em: {model_path}")

        mlp_step = final_model.named_steps["mlp"]
        target_results[target_key] = {
            "target_key": target_key,
            "target_label": cfg["label"],
            "unit": unit,
            "params": {
                **mlp_cfg,
                "hidden_layer_sizes": list(mlp_cfg["hidden_layer_sizes"]),
            },
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "residuals": residuals,
            "dist_test": dist_test,
            "train_metrics": train_metrics,
            "cv_summary": cv_summary,
            "test_metrics": test_metrics,
            "n_iter_full_train": int(mlp_step.n_iter_),
            "convergence_warnings_full_train": int(conv_warnings),
            "negative_predictions_test": int(np.sum(y_pred_test < 0.0)),
            "min_prediction_test": round(float(np.min(y_pred_test)), 4),
            "residuals_summary": {
                "mean": round(float(np.mean(residuals)), 4),
                "std": round(float(np.std(residuals)), 4),
                "median": round(float(np.median(residuals)), 4),
                "p05": round(float(np.percentile(residuals, 5)), 4),
                "p95": round(float(np.percentile(residuals, 95)), 4),
            },
            "shap_ranking": shap_ranking,
        }

    res_energy = target_results["energy_net_kwh"]
    res_soc = target_results["delta_soc_percent"]

    # Gera os 3 gráficos individuais do MLP (300 DPI) no mesmo padrão de XGBoost e Regressão Linear
    fig39_path = OUTPUT_FIGS_DIR / "39_mlp_real_vs_pred_energy.png"
    fig40_path = OUTPUT_FIGS_DIR / "40_mlp_residuals_distribution.png"
    fig41_path = OUTPUT_FIGS_DIR / "41_mlp_feature_importance_shap.png"

    plot_dual_real_vs_pred(
        res_energy,
        res_soc,
        fig39_path,
        suptitle="Desempenho Preditivo do MLP Regressor no Conjunto de Teste (Holdout 20%)",
        model_prep_phrase="pelo MLP",
        pred_key="y_pred_test",
        metrics_key="test_metrics",
        cv_key="cv_summary",
    )
    plot_dual_residuals_distribution(
        res_energy,
        res_soc,
        fig40_path,
        suptitle="Distribuição e Densidade dos Resíduos — MLP Regressor ($e = y_{real} - y_{pred}$)",
        residuals_key="residuals",
        headroom_scale=1.28,
    )
    plot_dual_shap_importance(
        res_energy,
        res_soc,
        fig41_path,
        suptitle="Atribuição de Importância das 12 Features Físicas — MLP Regressor (SHAP KernelExplainer)",
        ranking_key="shap_ranking",
        show_coef_sign=False,
    )

    if ARTIFACT_DIR.exists():
        for src_fig in (fig39_path, fig40_path, fig41_path):
            shutil.copy2(src_fig, ARTIFACT_DIR / src_fig.name)
            logger.info(f"Artefato copiado para brain: {ARTIFACT_DIR / src_fig.name}")

    # Salva JSON consolidado do MLP
    mlp_report = {
        "stage": "Etapa 3 - Avaliação final do MLP no Holdout (98 viagens) + SHAP KernelExplainer",
        "dataset_metadata": {
            "valid_trips": int(n_valid),
            "train_trips": int(len(train_idx)),
            "test_trips": int(n_holdout),
            "n_features": int(len(FEATURE_COLS)),
            "feature_columns": FEATURE_COLS,
        },
        "models": {
            k: {
                "label": r["target_label"],
                "unit": r["unit"],
                "params": r["params"],
                "train_in_sample": r["train_metrics"],
                "cv_5fold": r["cv_summary"],
                "holdout_test": r["test_metrics"],
                "n_iter_full_train": r["n_iter_full_train"],
                "convergence_warnings_full_train": r["convergence_warnings_full_train"],
                "negative_predictions_test": r["negative_predictions_test"],
                "min_prediction_test": r["min_prediction_test"],
                "residuals_summary": r["residuals_summary"],
                "shap_ranking": r["shap_ranking"],
            }
            for k, r in target_results.items()
        },
    }
    json_path = OUTPUT_RESULTS_DIR / "mlp_dual_model_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mlp_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório consolidado MLP salvo em: {json_path}")

    # Salva Ranking SHAP em CSV
    shap_rows = []
    for k, r in target_results.items():
        for item in r["shap_ranking"]:
            shap_rows.append({
                "target": k,
                "feature": item["feature"],
                "group": item["group"],
                "mean_abs_shap": item["mean_abs_shap"],
                "unit": r["unit"],
            })
    shap_csv_path = OUTPUT_RESULTS_DIR / "mlp_shap_rankings.csv"
    pd.DataFrame(shap_rows).to_csv(shap_csv_path, index=False)
    logger.info(f"Ranking SHAP MLP salvo em: {shap_csv_path}")

    # Consolida tabela comparativa dos 3 modelos (Regressão Linear, XGBoost e MLP)
    xgb_json = OUTPUT_RESULTS_DIR / "xgboost_dual_model_evaluation.json"
    lr_json = OUTPUT_RESULTS_DIR / "linear_regression_baseline_evaluation.json"
    if xgb_json.exists() and lr_json.exists():
        xgb_data = json.loads(xgb_json.read_text(encoding="utf-8"))
        lr_data = json.loads(lr_json.read_text(encoding="utf-8"))
        comp_rows = []
        for k in ("energy_net_kwh", "delta_soc_percent"):
            for m_label, src in (
                ("Regressão Linear (Baseline)", lr_data["models"][k]),
                ("XGBoost", xgb_data["models"][k]),
                ("MLP", mlp_report["models"][k]),
            ):
                comp_rows.append({
                    "target": k,
                    "model": m_label,
                    "cv_r2": src["cv_5fold"]["mean_r2"],
                    "cv_r2_std": src["cv_5fold"]["std_r2"],
                    "cv_mae": src["cv_5fold"]["mean_mae"],
                    "cv_rmse": src["cv_5fold"]["mean_rmse"],
                    "test_r2": src["holdout_test"]["r2"],
                    "test_mae": src["holdout_test"]["mae"],
                    "test_rmse": src["holdout_test"]["rmse"],
                    "test_mape_percent": src["holdout_test"]["mape_percent"],
                    "test_mae_100km": src["holdout_test"]["mae_100km"],
                })
        comp_csv_path = OUTPUT_RESULTS_DIR / "phase2_three_models_comparison.csv"
        pd.DataFrame(comp_rows).to_csv(comp_csv_path, index=False)
        logger.info(f"Tabela comparativa 3 modelos salva em: {comp_csv_path}")

    generate_three_models_comparison_figures()
    return mlp_report


def plot_three_models_bars_cv_vs_test(
    lr_data: dict, xgb_data: dict, mlp_data: dict, save_path: Path
) -> None:
    """Opção A: Painel 2×2 de barras agrupadas (CV 5-fold vs. Teste Holdout) para R² e MAE."""
    sns.set_theme(style="ticks", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=300)

    fig.suptitle(
        "Comparação dos Modelos da Fase 2: Validação Cruzada (N = 388) vs. Teste Final (N = 98)",
        fontsize=13.5,
        fontweight="bold",
        y=0.975,
        color="#0F172A",
    )

    model_specs = [
        ("Regressão Linear\n(Baseline)", lr_data, "#FDBA74", "#EA580C"),
        ("XGBoost", xgb_data, "#93C5FD", "#1D4ED8"),
        ("MLP", mlp_data, "#6EE7B7", "#047857"),
    ]
    cols_cfg = [
        (0, "energy_net_kwh", "Energia Líquida (kWh)", "kWh", 4, 0.84, 1.03),
        (1, "delta_soc_percent", "Queda de Carga — ΔSOC (%)", "%", 2, 0.55, 0.98),
    ]

    x = np.arange(len(model_specs))
    width = 0.31
    gap = 0.03

    for col_idx, target_key, col_title, unit, mae_dec, r2_ymin, r2_ymax in cols_cfg:
        cv_r2 = [src["models"][target_key]["cv_5fold"]["mean_r2"] for _, src, _, _ in model_specs]
        cv_r2_std = [src["models"][target_key]["cv_5fold"]["std_r2"] for _, src, _, _ in model_specs]
        test_r2 = [src["models"][target_key]["holdout_test"]["r2"] for _, src, _, _ in model_specs]

        cv_mae = [src["models"][target_key]["cv_5fold"]["mean_mae"] for _, src, _, _ in model_specs]
        test_mae = [src["models"][target_key]["holdout_test"]["mae"] for _, src, _, _ in model_specs]
        test_mape = [src["models"][target_key]["holdout_test"]["mape_percent"] for _, src, _, _ in model_specs]

        light_colors = [c_light for _, _, c_light, _ in model_specs]
        dark_colors = [c_dark for _, _, _, c_dark in model_specs]
        labels = [lbl for lbl, _, _, _ in model_specs]

        # Linha 1: R² (CV vs. Teste)
        ax_r2 = axes[0, col_idx]
        b_cv = ax_r2.bar(
            x - width / 2 - gap, cv_r2, width=width, color=light_colors,
            edgecolor="#334155", linewidth=0.9, hatch="//", zorder=3,
        )
        b_te = ax_r2.bar(
            x + width / 2 + gap, test_r2, width=width, color=dark_colors,
            edgecolor="#0F172A", linewidth=0.9, zorder=3,
        )
        ax_r2.errorbar(
            x - width / 2 - gap, cv_r2, yerr=cv_r2_std, fmt="none",
            ecolor="#475569", elinewidth=1.1, capsize=3.5, zorder=4,
        )

        for bar, val, std in zip(b_cv, cv_r2, cv_r2_std):
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.012 * (r2_ymax - r2_ymin),
                _fmt_pt(val, 4),
                ha="center", va="bottom", fontsize=8.8, color="#334155", fontweight="semibold",
            )
        for bar, val in zip(b_te, test_r2):
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015 * (r2_ymax - r2_ymin),
                _fmt_pt(val, 4),
                ha="center", va="bottom", fontsize=9.2, color="#0F172A", fontweight="bold",
            )

        ax_r2.set_ylim(r2_ymin, r2_ymax)
        if col_idx == 0:
            ax_r2.set_yticks([0.85, 0.88, 0.91, 0.94, 0.97, 1.00])
        else:
            ax_r2.set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(labels, fontsize=9.5, color="#1E293B")
        ax_r2.set_title(col_title, fontsize=11.8, fontweight="bold", pad=10, color="#0F172A")
        ax_r2.set_ylabel("R² (maior = melhor)" if col_idx == 0 else "", fontsize=10, color="#334155")
        ax_r2.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax_r2)

        # Linha 2: MAE (CV vs. Teste) + MAPE do Teste
        ax_mae = axes[1, col_idx]
        b_cv_m = ax_mae.bar(
            x - width / 2 - gap, cv_mae, width=width, color=light_colors,
            edgecolor="#334155", linewidth=0.9, hatch="//", zorder=3,
        )
        b_te_m = ax_mae.bar(
            x + width / 2 + gap, test_mae, width=width, color=dark_colors,
            edgecolor="#0F172A", linewidth=0.9, zorder=3,
        )

        max_mae = max(max(cv_mae), max(test_mae))
        for bar, val in zip(b_cv_m, cv_mae):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.025 * max_mae,
                f"{_fmt_pt(val, mae_dec)}\n{unit}",
                ha="center", va="bottom", fontsize=8.4, color="#334155", fontweight="semibold",
            )
        for bar, val, mp in zip(b_te_m, test_mae, test_mape):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.025 * max_mae,
                f"{_fmt_pt(val, mae_dec)} {unit}\nMAPE {_fmt_pt(mp, 1)}%",
                ha="center", va="bottom", fontsize=8.0, color="#0F172A", fontweight="bold",
            )

        ax_mae.set_ylim(0.0, max_mae * 1.32)
        ax_mae.set_xticks(x)
        ax_mae.set_xticklabels(labels, fontsize=9.5, color="#1E293B")
        ax_mae.set_ylabel("MAE (menor = melhor)" if col_idx == 0 else "", fontsize=10, color="#334155")
        ax_mae.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax_mae)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#E2E8F0", edgecolor="#334155", hatch="//", label="CV 5-Fold no Treino (N = 388, barras de erro = ±1 DP)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#475569", edgecolor="#0F172A", label="Teste Final Holdout (N = 98 viagens inéditas)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2, frameon=True,
        facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5, bbox_to_anchor=(0.5, 0.935),
    )
    fig.text(
        0.5, 0.012,
        "Referência piso Dummy (Média no teste): Energia Líquida R² = −0,0145 (MAE = 0,4036 kWh) | ΔSOC R² = −0,0020 (MAE = 2,69 %).",
        ha="center", fontsize=8.8, color="#475569", style="italic",
    )

    plt.tight_layout(rect=[0.02, 0.035, 0.98, 0.895])
    plt.subplots_adjust(hspace=0.28, wspace=0.18)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura Opção A (barras CV vs Teste) salva: {save_path}")


def plot_three_models_scatter_and_bars(
    preds_by_target: dict,
    lr_data: dict,
    xgb_data: dict,
    mlp_data: dict,
    save_path: Path,
) -> None:
    """Opção B: Painel 3×2 (Dispersão Real vs. Predito + Barras de R² e MAE no Teste Holdout)."""
    sns.set_theme(style="ticks", font_scale=1.0)
    fig, axes = plt.subplots(
        3, 2, figsize=(13.2, 12.6), dpi=300,
        gridspec_kw={"height_ratios": [1.6, 0.82, 0.85]},
    )

    fig.suptitle(
        "Comparação dos Três Modelos no Conjunto de Teste Final (N = 98 viagens)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
        color="#0F172A",
    )

    c_lr = "#EA580C"
    c_xgb = "#1D4ED8"
    c_mlp = "#047857"

    cols_cfg = [
        (0, "energy_net_kwh", "Energia Líquida (kWh)", "kWh", -0.12, 4, 0.85, 1.02),
        (1, "delta_soc_percent", "Queda de Carga — ΔSOC (%)", "%", -1.1, 2, 0.55, 0.92),
    ]

    for col_idx, target_key, col_title, unit, y_floor, mae_dec, r2_ymin, r2_ymax in cols_cfg:
        p = preds_by_target[target_key]
        y_real = p["y_test"]
        y_lr = p["lr_pred"]
        y_xgb = p["xgb_pred"]
        y_mlp = p["mlp_pred"]

        # Linha 1: Dispersão Real vs. Predito
        ax_sc = axes[0, col_idx]
        hi = max(float(np.max(y_real)), float(np.max(y_lr)), float(np.max(y_xgb)), float(np.max(y_mlp))) * 1.08

        ax_sc.axhline(0.0, color="#CBD5E1", linestyle="-", linewidth=1.0, zorder=1)
        ax_sc.plot([0.0, hi], [0.0, hi], color="#0F172A", linestyle="--", linewidth=1.4, label="Ideal (y = x)", zorder=2)

        ax_sc.scatter(
            y_real, y_lr, facecolors="none", edgecolors=c_lr,
            s=40, linewidths=1.25, alpha=0.82, label="Regressão Linear (Baseline)", zorder=3,
        )
        ax_sc.scatter(
            y_real, y_xgb, marker="^", color=c_xgb, edgecolors="white",
            s=36, linewidths=0.4, alpha=0.82, label="XGBoost", zorder=4,
        )
        ax_sc.scatter(
            y_real, y_mlp, marker="o", color=c_mlp, edgecolors="white",
            s=34, linewidths=0.45, alpha=0.90, label="MLP", zorder=5,
        )

        ax_sc.set_xlim(-0.04 * hi, hi)
        ax_sc.set_ylim(y_floor, hi)
        ax_sc.set_title(col_title, fontsize=12, fontweight="bold", pad=10, color="#0F172A")
        ax_sc.set_xlabel(f"Valor real ({unit})", fontsize=10, color="#334155")
        ax_sc.set_ylabel(f"Valor predito ({unit})" if col_idx == 0 else "", fontsize=10, color="#334155")
        ax_sc.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax_sc)

        # Métricas do Teste para Barras
        models_src = [
            ("Regressão Linear\n(Baseline)", lr_data["models"][target_key]["holdout_test"], c_lr),
            ("XGBoost", xgb_data["models"][target_key]["holdout_test"], c_xgb),
            ("MLP", mlp_data["models"][target_key]["holdout_test"], c_mlp),
        ]
        x_pos = np.arange(len(models_src))
        m_labels = [lbl for lbl, _, _ in models_src]
        b_colors = [c for _, _, c in models_src]
        r2_vals = [m["r2"] for _, m, _ in models_src]
        mae_vals = [m["mae"] for _, m, _ in models_src]
        mape_vals = [m["mape_percent"] for _, m, _ in models_src]

        # Linha 2: R² no Teste
        ax_r2 = axes[1, col_idx]
        bars_r2 = ax_r2.bar(x_pos, r2_vals, width=0.5, color=b_colors, edgecolor="#0F172A", linewidth=0.8, zorder=3)
        for bar, val in zip(bars_r2, r2_vals):
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.025 * (r2_ymax - r2_ymin),
                _fmt_pt(val, 4),
                ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#0F172A",
            )
        ax_r2.set_ylim(r2_ymin, r2_ymax)
        ax_r2.set_xticks(x_pos)
        ax_r2.set_xticklabels(m_labels, fontsize=9.5, color="#1E293B")
        ax_r2.set_ylabel("R² no Teste (maior = melhor)" if col_idx == 0 else "", fontsize=9.8, color="#334155")
        ax_r2.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax_r2)

        # Linha 3: MAE no Teste (+ MAPE)
        ax_mae = axes[2, col_idx]
        bars_mae = ax_mae.bar(x_pos, mae_vals, width=0.5, color=b_colors, edgecolor="#0F172A", linewidth=0.8, zorder=3)
        max_mae = max(mae_vals)
        for bar, mv, mp in zip(bars_mae, mae_vals, mape_vals):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                mv + 0.03 * max_mae,
                f"{_fmt_pt(mv, mae_dec)} {unit}\n(MAPE {_fmt_pt(mp, 2)}%)",
                ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0F172A",
            )
        ax_mae.set_ylim(0.0, max_mae * 1.32)
        ax_mae.set_xticks(x_pos)
        ax_mae.set_xticklabels(m_labels, fontsize=9.5, color="#1E293B")
        ax_mae.set_ylabel("MAE no Teste (menor = melhor)" if col_idx == 0 else "", fontsize=9.8, color="#334155")
        ax_mae.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
        sns.despine(ax=ax_mae)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=True, framealpha=0.95,
        facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5, bbox_to_anchor=(0.5, 0.947),
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.908])
    plt.subplots_adjust(hspace=0.33, wspace=0.18)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figura Opção B (dispersão + barras) salva: {save_path}")


def generate_three_models_comparison_figures() -> None:
    """Gera os dois gráficos comparativos (Opção A e Opção B) dos 3 modelos finais."""
    lr_json = OUTPUT_RESULTS_DIR / "linear_regression_baseline_evaluation.json"
    xgb_json = OUTPUT_RESULTS_DIR / "xgboost_dual_model_evaluation.json"
    mlp_json = OUTPUT_RESULTS_DIR / "mlp_dual_model_evaluation.json"
    for p in (lr_json, xgb_json, mlp_json):
        if not p.exists():
            raise FileNotFoundError(f"Relatório necessário não encontrado: {p}")

    lr_data = json.loads(lr_json.read_text(encoding="utf-8"))
    xgb_data = json.loads(xgb_json.read_text(encoding="utf-8"))
    mlp_data = json.loads(mlp_json.read_text(encoding="utf-8"))

    # Carrega previsões no mesmo conjunto de teste (98 viagens) usando modelos salvos + LinearRegression
    df_valid, train_idx, test_idx, _, _ = _load_train_split()
    X_train = df_valid[FEATURE_COLS].iloc[train_idx].copy()
    X_test = df_valid[FEATURE_COLS].iloc[test_idx].copy()

    preds_by_target = {}
    for target_key in ("energy_net_kwh", "delta_soc_percent"):
        cfg = TARGET_CONFIGS[target_key]
        y = df_valid[cfg["target_col"]].to_numpy(dtype=np.float64)
        y_train, y_test = y[train_idx], y[test_idx]

        lr_model = LinearRegression().fit(X_train, y_train)
        xgb_bundle = joblib.load(OUTPUT_MODELS_DIR / cfg["model_file"])
        mlp_bundle = joblib.load(OUTPUT_MODELS_DIR / MLP_FINAL_CONFIGS[target_key]["model_file"])

        preds_by_target[target_key] = {
            "y_test": y_test,
            "lr_pred": lr_model.predict(X_test),
            "xgb_pred": xgb_bundle["model"].predict(X_test),
            "mlp_pred": mlp_bundle["model"].predict(X_test),
        }

    fig42_path = OUTPUT_FIGS_DIR / "42_comparacao_3_modelos_barras_cv_teste.png"
    fig43_path = OUTPUT_FIGS_DIR / "43_comparacao_3_modelos_dispersao_barras.png"

    plot_three_models_bars_cv_vs_test(lr_data, xgb_data, mlp_data, fig42_path)
    plot_three_models_scatter_and_bars(preds_by_target, lr_data, xgb_data, mlp_data, fig43_path)

    if ARTIFACT_DIR.exists():
        for src_fig in (fig42_path, fig43_path):
            shutil.copy2(src_fig, ARTIFACT_DIR / src_fig.name)
            logger.info(f"Artefato copiado para brain: {ARTIFACT_DIR / src_fig.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline em etapas do MLP Regressor (Fase 2).")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "2b", "3", "compare"],
        default="3",
        help="Etapa a executar (1, 2, 2b, 3 ou compare).",
    )
    args = parser.parse_args()
    if args.stage == "1":
        run_mlp_stage1_diagnostic()
    elif args.stage == "2":
        run_mlp_stage2_cv_tuning()
    elif args.stage == "2b":
        run_mlp_stage2b_refinement()
    elif args.stage == "compare":
        generate_three_models_comparison_figures()
    else:
        run_mlp_stage3_final_evaluation()
