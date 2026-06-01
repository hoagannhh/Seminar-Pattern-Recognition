# ============================================================
#  src/evaluation.py  —  wikicities-population
#
#  Các hàm đánh giá model: RMSE, MAE, R², feature importance.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Metrics ─────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series,
                cv: int = 5, seed: int = 42) -> dict:
    """
    Chạy cross-validation và trả về dict chứa mean/std của RMSE, MAE, R².

    Returns
    -------
    dict với keys: rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    rmse_scores = -cross_val_score(model, X, y, cv=kf,
                                   scoring="neg_root_mean_squared_error")
    mae_scores  = -cross_val_score(model, X, y, cv=kf,
                                   scoring="neg_mean_absolute_error")
    r2_scores   =  cross_val_score(model, X, y, cv=kf, scoring="r2")

    return {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std":  rmse_scores.std(),
        "mae_mean":  mae_scores.mean(),
        "mae_std":   mae_scores.std(),
        "r2_mean":   r2_scores.mean(),
        "r2_std":    r2_scores.std(),
    }


def print_cv_results(results: dict, model_name: str = "Model") -> None:
    """In kết quả CV theo định dạng dễ đọc."""
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  RMSE : {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"  MAE  : {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"  R²   : {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")


def compare_models(models: dict, X: pd.DataFrame, y: pd.Series,
                   cv: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    So sánh nhiều model cùng lúc.

    Parameters
    ----------
    models : dict  — {tên_model: sklearn_estimator}

    Returns
    -------
    pd.DataFrame với các metrics cho từng model, sắp xếp theo RMSE tăng dần.
    """
    rows = []
    for name, model in models.items():
        res = evaluate_cv(model, X, y, cv=cv, seed=seed)
        rows.append({"Model": name, **res})
        print_cv_results(res, name)
    df = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    return df


# ── Feature Importance ───────────────────────────────────────

def get_feature_importance(model, feature_names: list[str],
                           importance_type: str = "auto") -> pd.Series:
    """
    Trích xuất feature importance từ tree-based model.
    Hỗ trợ sklearn (feature_importances_) và XGBoost/LightGBM.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "get_score"):           # XGBoost Booster
        scores = model.get_score(importance_type="gain")
        imp = np.array([scores.get(f, 0) for f in feature_names])
    else:
        raise ValueError("Model không hỗ trợ feature importance.")
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, top_n: int = 20,
                            title: str = "Feature Importance",
                            save_path: Optional[Path] = None) -> None:
    """Vẽ bar chart feature importance (top_n features)."""
    data = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    data.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── Stability Analysis ───────────────────────────────────────

def stability_cv(model, X: pd.DataFrame, y: pd.Series,
                 n_runs: int = 10, cv: int = 5,
                 seed_start: int = 0) -> pd.DataFrame:
    """
    Chạy CV nhiều lần với các random seed khác nhau để đo tính ổn định.

    Returns
    -------
    pd.DataFrame với cột rmse cho mỗi run × fold.
    """
    records = []
    for run in range(n_runs):
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed_start + run)
        scores = -cross_val_score(model, X, y, cv=kf,
                                  scoring="neg_root_mean_squared_error")
        for fold, s in enumerate(scores):
            records.append({"run": run, "fold": fold, "rmse": s})
    return pd.DataFrame(records)


def plot_stability(stability_df: pd.DataFrame,
                   label: str = "Model",
                   ax: Optional[plt.Axes] = None,
                   color: str = "steelblue") -> plt.Axes:
    """
    Boxplot phân phối RMSE qua các run để minh hoạ tính ổn định.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    rmse_per_run = stability_df.groupby("run")["rmse"].mean()
    ax.boxplot(rmse_per_run, vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6))
    ax.set_title(f"Stability — {label}")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_xticks([1]); ax.set_xticklabels([label])
    return ax


def compare_stability(results: dict[str, pd.DataFrame],
                      save_path: Optional[Path] = None) -> None:
    """
    So sánh stability của nhiều feature set / model.

    Parameters
    ----------
    results : dict  — {tên: stability_df từ stability_cv()}
    """
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 2), 5))
    colors = plt.cm.tab10.colors
    data_list, labels = [], []

    for i, (name, df) in enumerate(results.items()):
        rmse_per_run = df.groupby("run")["rmse"].mean().values
        data_list.append(rmse_per_run)
        labels.append(name)

    bp = ax.boxplot(data_list, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Stability Comparison — RMSE across 10 random seeds")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    print("\nSummary:")
    for name, df in results.items():
        per_run = df.groupby("run")["rmse"].mean()
        print(f"  {name:35s}: {per_run.mean():.4f} ± {per_run.std():.4f}")
