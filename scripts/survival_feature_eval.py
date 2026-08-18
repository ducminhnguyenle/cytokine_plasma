"""
survival_feature_eval.py

Reusable pipeline for training/evaluating a Random Survival Forest (RSF)
across multiple candidate feature subsets, with per-cohort standardization,
C-index scoring, time-dependent AUC, and plotting.

The scaling logic intentionally mirrors the original notebook exactly:
a single StandardScaler instance is fit/refit per (cohort, column) group
during training, and its *final* fitted state is what gets applied via
`.transform` on the test set. This preserves original behavior 1:1.

Typical usage
-------------
    from survival_feature_eval import run_all_feature_sets

    cols_01 = ["ss"]
    cols_02 = ["il6", "il8", "il10", "tnfalpha"]
    cols_03 = ["ss", "il6", "il8", "il10", "tnfalpha"]
    cols_04 = ["ss", "il6", "il10", "tnfalpha"]
    cols_05 = ["ss", "il8", "il10", "tnfalpha"]
    cols_06 = ["ss", "il10", "tnfalpha"]

    feature_sets = {
        "set_01": cols_01,
        "set_02": cols_02,
        "set_03": cols_03,
        "set_04": cols_04,
        "set_05": cols_05,
        "set_06": cols_06,
    }

    results = run_all_feature_sets(
        feature_sets=feature_sets,
        cytokines_df=cytokines_df,
        numeric_features=numeric_features,
        y=y,
        times=times,
    )

    for name, res in results.items():
        print(name, res.c_index, res.mean_auc)

    # Or run/inspect just one set:
    result_02 = run_feature_set("set_02", cols_02, cytokines_df, numeric_features, y, times)
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc


@dataclass
class FeatureSetResult:
    """Container holding everything produced for one feature subset run."""

    name: str
    selected_features: List[str]
    c_index: float
    mean_auc: float
    auc_by_time: np.ndarray
    model: RandomSurvivalForest
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


# --------------------------------------------------------------------------- #
# Individual pipeline steps (each usable standalone)
# --------------------------------------------------------------------------- #

def get_selected_features(numeric_features: pd.Index, cols: Iterable[str]) -> List[str]:
    """Intersect the requested column list with available numeric features.

    Returns a plain list (not an Index) so it can be safely concatenated
    elsewhere, and is robust to `cols` containing names absent from the data.
    """
    selected = numeric_features.intersection(list(cols))
    return selected.tolist()


def split_data(
    df: pd.DataFrame,
    y,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 2024,
    stratify_col: str = "event",
):
    """Stratified train/test split, identical to the original notebook call."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y[stratify_col],
    )
    return X_train, X_test, y_train, y_test


def scale_by_cohort(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: List[str],
    cohort_col: str = "cohort",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cohort StandardScaler transform, matching original logic exactly.

    NOTE: a single `StandardScaler` instance is reused across every
    (cohort, feature) group during `.fit_transform` on the training set, so
    by the time it is applied (`.transform`) to the test set it only carries
    the parameters from the *last* group/column it saw while fitting. This
    matches the behavior of the original code verbatim.
    """
    sc = StandardScaler()

    X_train_transformed = (
        X_train.groupby(cohort_col)[selected_features]
        .transform(lambda x: sc.fit_transform(x.to_numpy().reshape(-1, 1)).ravel())
    )

    X_test_transformed = (
        X_test.groupby(cohort_col)[selected_features]
        .transform(lambda x: sc.transform(x.to_numpy().reshape(-1, 1)).ravel())
    )

    return X_train_transformed, X_test_transformed


def train_rsf(
    X_train: pd.DataFrame,
    y_train,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    random_state: int = 2026,
) -> RandomSurvivalForest:
    """Fit a Random Survival Forest with the original default hyperparameters."""
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    rsf.fit(X_train, y_train)
    return rsf


def evaluate_rsf(
    rsf: RandomSurvivalForest,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    times,
) -> Tuple[float, np.ndarray, float]:
    """Compute C-index and time-dependent cumulative/dynamic AUC."""
    c_index = rsf.score(X_test, y_test)
    risk_scores = rsf.predict(X_test)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    return c_index, auc, mean_auc


def plot_auc(
    times,
    auc,
    mean_auc: float,
    label_prefix: str,
    title_suffix: str,
    ax: Optional[plt.Axes] = None,
):
    """Plot time-dependent AUC; reuses an existing axis if provided so
    multiple feature sets can be overlaid on one figure."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(times, auc, "o-", label=f"{label_prefix} (mean AUC = {mean_auc:.3f})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("time-dependent AUC")
    ax.set_title(f"Time-dependent AUC by {title_suffix}", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    return ax


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_feature_set(
    name: str,
    cols: Iterable[str],
    cytokines_df: pd.DataFrame,
    numeric_features: pd.Index,
    y,
    times,
    cohort_col: str = "cohort",
    test_size: float = 0.2,
    split_random_state: int = 2024,
    rsf_kwargs: Optional[dict] = None,
    plot: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    verbose: bool = True,
) -> FeatureSetResult:
    """Run the full select -> split -> scale -> train -> evaluate -> plot
    pipeline for a single named feature subset.

    Robust to feature subsets that only partially (or don't at all) overlap
    with `numeric_features` -- raises a clear error instead of silently
    training on zero features.
    """
    rsf_kwargs = rsf_kwargs or {}

    selected_features = get_selected_features(numeric_features, cols)
    if not selected_features:
        raise ValueError(
            f"No overlap between numeric_features and requested cols for "
            f"'{name}': {list(cols)}"
        )

    if verbose:
        print(f"[{name}] Selected features for ML model: {selected_features}")

    feature_cols = selected_features + [cohort_col]

    X_train, X_test, y_train, y_test = split_data(
        cytokines_df,
        y,
        feature_cols,
        test_size=test_size,
        random_state=split_random_state,
    )
    if verbose:
        print(f"[{name}] Training set size: {X_train.shape[0]} samples")
        print(f"[{name}] Test set size: {X_test.shape[0]} samples")

    X_train_transformed, X_test_transformed = scale_by_cohort(
        X_train, X_test, selected_features, cohort_col=cohort_col
    )

    rsf = train_rsf(X_train_transformed, y_train, **rsf_kwargs)

    c_index, auc, mean_auc = evaluate_rsf(
        rsf, X_test_transformed, y_train, y_test, times
    )
    if verbose:
        print(f"[{name}] C-index: {round(c_index, 3)}")
        print(f"[{name}] Mean AUC: {mean_auc:.3f}")

    if plot:
        plot_auc(times, auc, mean_auc, label_prefix="RSF", title_suffix=name, ax=ax)
        if ax is None and show:
            plt.show()

    return FeatureSetResult(
        name=name,
        selected_features=selected_features,
        c_index=c_index,
        mean_auc=mean_auc,
        auc_by_time=auc,
        model=rsf,
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
    )


def run_all_feature_sets(
    feature_sets: Dict[str, Iterable[str]],
    cytokines_df: pd.DataFrame,
    numeric_features: pd.Index,
    y,
    times,
    overlay_plot: bool = False,
    **kwargs,
) -> Dict[str, FeatureSetResult]:
    """Run `run_feature_set` for every {name: cols} pair.

    Parameters
    ----------
    overlay_plot : if True, all feature sets are plotted on a single shared
        axis instead of one figure per set (useful for comparing AUC curves).
    **kwargs : forwarded to `run_feature_set` (e.g. rsf_kwargs, test_size,
        cohort_col, verbose, plot).
    """
    results: Dict[str, FeatureSetResult] = {}

    ax = None
    if overlay_plot:
        _, ax = plt.subplots()

    for name, cols in feature_sets.items():
        results[name] = run_feature_set(
            name=name,
            cols=cols,
            cytokines_df=cytokines_df,
            numeric_features=numeric_features,
            y=y,
            times=times,
            ax=ax,
            show=False,
            **kwargs,
        )

    if overlay_plot:
        ax.set_title("Time-dependent AUC across feature sets", fontsize=16, fontweight="bold")
        plt.show()
    elif not kwargs.get("plot", True):
        pass  # nothing to show
    else:
        # individual figures were already shown inside run_feature_set()
        pass

    return results


def summarize_results(results: Dict[str, FeatureSetResult]) -> pd.DataFrame:
    """Collapse results into a tidy comparison table: name, n_features, C-index, mean AUC."""
    rows = [
        {
            "feature_set": name,
            "n_features": len(res.selected_features),
            "features": ", ".join(res.selected_features),
            "c_index": round(res.c_index, 3),
            "mean_auc": round(res.mean_auc, 3),
        }
        for name, res in results.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
