from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .feature_engineering import DesignMatrix, fit_design_matrix, transform_design_matrix

SEED = 42


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ndcg_at_k(y_true_log: np.ndarray, y_pred_log: np.ndarray, k: int = 20) -> float:
    if len(y_true_log) == 0:
        return 0.0
    k = min(k, len(y_true_log))
    order = np.argsort(-y_pred_log)[:k]
    ideal = np.argsort(-y_true_log)[:k]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    gains = np.expm1(np.clip(y_true_log, 0, 15))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def metric_record(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    error = y_true_log - y_pred_log
    raw = np.expm1(y_true_log)
    top_n = max(1, int(np.ceil(0.10 * len(raw))))
    selected = np.argsort(-y_pred_log)[:top_n]
    overall = float(np.mean(raw))
    return {
        "mae_log": float(np.mean(np.abs(error))),
        "rmse_log": float(np.sqrt(np.mean(error ** 2))),
        "spearman": spearman(y_true_log, y_pred_log),
        "top_decile_lift": float(np.mean(raw[selected]) / overall) if overall > 0 else 0.0,
        "ndcg_at_20": ndcg_at_k(y_true_log, y_pred_log, 20),
    }


class DummyMedian:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "DummyMedian":
        self.value = float(np.median(y))
        self.n_features = x.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)

    def importances(self, names: list[str]) -> pd.Series:
        return pd.Series(0.0, index=names)

    def contributions(self, x: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(x), self.value), np.zeros((len(x), len(names)))


class RidgeRegressor:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.coef_[0] + x @ self.coef_[1:]

    def importances(self, names: list[str]) -> pd.Series:
        values = np.abs(self.coef_[1:])
        return pd.Series(values / values.sum() if values.sum() else values, index=names)

    def contributions(self, x: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(x), self.coef_[0]), x * self.coef_[1:]


class GradientBoostedStumps:
    """Deterministic NumPy fallback used because CatBoost/LightGBM are unavailable."""

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.08, quantiles: int = 9):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.quantiles = quantiles

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedStumps":
        self.base_ = float(np.mean(y))
        pred = np.full(len(y), self.base_)
        self.stumps_: list[tuple[int, float, float, float, float]] = []
        self.gains_: dict[int, float] = {}
        for _ in range(self.n_estimators):
            residual = y - pred
            base_sse = float(np.sum((residual - residual.mean()) ** 2))
            best: tuple[float, int, float, float, float] | None = None
            for feature in range(x.shape[1]):
                values = x[:, feature]
                thresholds = np.unique(np.quantile(values, np.linspace(0.1, 0.9, self.quantiles)))
                for threshold in thresholds:
                    left = values <= threshold
                    if left.sum() < 8 or (~left).sum() < 8:
                        continue
                    lv = float(residual[left].mean())
                    rv = float(residual[~left].mean())
                    sse = float(np.sum((residual[left] - lv) ** 2) + np.sum((residual[~left] - rv) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, feature, float(threshold), lv, rv)
            if best is None:
                break
            sse, feature, threshold, lv, rv = best
            lv *= self.learning_rate
            rv *= self.learning_rate
            self.stumps_.append((feature, threshold, lv, rv, max(0.0, base_sse - sse)))
            self.gains_[feature] = self.gains_.get(feature, 0.0) + max(0.0, base_sse - sse)
            pred += np.where(x[:, feature] <= threshold, lv, rv)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(len(x), self.base_)
        for feature, threshold, lv, rv, _ in self.stumps_:
            pred += np.where(x[:, feature] <= threshold, lv, rv)
        return pred

    def importances(self, names: list[str]) -> pd.Series:
        values = np.array([self.gains_.get(i, 0.0) for i in range(len(names))])
        return pd.Series(values / values.sum() if values.sum() else values, index=names)

    def contributions(self, x: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        contrib = np.zeros((len(x), len(names)))
        for feature, threshold, lv, rv, _ in self.stumps_:
            contrib[:, feature] += np.where(x[:, feature] <= threshold, lv, rv)
        return np.full(len(x), self.base_), contrib


MODELS = {
    "DummyMedian": lambda: DummyMedian(),
    "Ridge": lambda: RidgeRegressor(alpha=10.0),
    "GradientBoostedStumpsFallback": lambda: GradientBoostedStumps(),
}


def group_folds(groups: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = groups.value_counts().sort_values(ascending=False)
    buckets: list[list[Any]] = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for group, count in counts.items():
        idx = int(np.argmin(sizes))
        buckets[idx].append(group)
        sizes[idx] += int(count)
    folds = []
    for bucket in buckets:
        valid = groups.isin(bucket).to_numpy()
        folds.append((np.where(~valid)[0], np.where(valid)[0]))
    return folds


def leave_one_group_out(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.where(groups.ne(group).to_numpy())[0], np.where(groups.eq(group).to_numpy())[0])
        for group in sorted(groups.unique())
    ]


def _evaluate_scheme(
    frame: pd.DataFrame, descriptive: bool, folds: Iterable[tuple[np.ndarray, np.ndarray]],
    country: str, experiment: str, scheme: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = {name: np.full(len(frame), np.nan) for name in MODELS}
    for train_idx, valid_idx in folds:
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        design = fit_design_matrix(train, descriptive=descriptive)
        x_train = design.x
        x_valid = transform_design_matrix(valid, design)
        y_train = np.log1p(train["monthly_sold_value"].to_numpy(float))
        for name, factory in MODELS.items():
            model = factory().fit(x_train, y_train)
            predictions[name][valid_idx] = model.predict(x_valid)
    metrics: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    y = np.log1p(frame["monthly_sold_value"].to_numpy(float))
    for model_name, pred in predictions.items():
        if np.isnan(pred).any():
            raise RuntimeError(f"Missing out-of-fold predictions for {country}/{experiment}/{model_name}/{scheme}")
        record = {
            "country_code": country, "experiment": experiment, "model": model_name,
            "validation_scheme": scheme, "n_rows": len(frame), **metric_record(y, pred),
        }
        metrics.append(record)
        for idx, value in enumerate(pred):
            pred_rows.append({
                "country_code": country, "experiment": experiment, "model": model_name,
                "validation_scheme": scheme, "row_index": int(frame.index[idx]),
                "actual_log_target": y[idx], "predicted_log_target": value,
            })
    return metrics, pred_rows


def _friendly_feature(name: str) -> str:
    mapping = {
        "log1p_liked_count": "listing likes", "log1p_rating_count": "rating volume",
        "price_pct_country_category": "peer-relative price", "price_pct_shop": "shop-relative price",
        "displayed_discount_pct": "displayed discount depth", "discount_pct_peer": "peer-relative discount",
        "is_promoted": "current promotion status", "promotion_mechanism_count": "promotion mechanism count",
        "log1p_follower_count": "shop followers", "rating_star": "shop rating",
        "shop_assortment_size": "shop assortment size", "log1p_history_sold": "historical sold-value proxy",
        "sold_pct_peer": "historical sold-value peer percentile",
    }
    return mapping.get(name, name.replace("_", " ").replace("=", ": "))


def run_model_experiments(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling = frame.loc[frame["monthly_sold_value"].notna() & frame["monthly_sold_value"].ge(0)].copy()
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for country, country_frame in modeling.groupby("country_code"):
        country_frame = country_frame.copy()
        groups = country_frame["shop_id"]
        for experiment, descriptive in [
            ("actionable_context", False), ("descriptive_leakage_prone", True)
        ]:
            for scheme, folds in [
                ("group_5_fold", group_folds(groups, min(5, groups.nunique()))),
                ("leave_one_shop_out", leave_one_group_out(groups)),
            ]:
                metrics, preds = _evaluate_scheme(
                    country_frame, descriptive, folds, country, experiment, scheme
                )
                metric_rows.extend(metrics)
                pred_rows.extend(preds)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(pred_rows)

    importance_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    actionable_metrics = metrics.loc[
        (metrics.experiment == "actionable_context")
        & (metrics.validation_scheme == "group_5_fold")
        & (metrics.model != "DummyMedian")
    ]
    for country, country_frame in modeling.groupby("country_code"):
        best_name = actionable_metrics.loc[actionable_metrics.country_code == country].sort_values(
            ["rmse_log", "ndcg_at_20"], ascending=[True, False]
        ).iloc[0].model
        design = fit_design_matrix(country_frame, descriptive=False)
        model = MODELS[best_name]().fit(
            design.x, np.log1p(country_frame["monthly_sold_value"].to_numpy(float))
        )
        imp = model.importances(design.feature_names).sort_values(ascending=False)
        for feature, value in imp.items():
            importance_rows.append({
                "country_code": country, "experiment": "actionable_context",
                "model": best_name, "feature": feature, "feature_label": _friendly_feature(feature),
                "importance": float(value),
            })
        representatives = country_frame.sort_values("opportunity_score", ascending=False).head(5)
        x_rep = transform_design_matrix(representatives, design)
        intercept, contributions = model.contributions(x_rep, design.feature_names)
        predictions_rep = model.predict(x_rep)
        for i, (idx, row) in enumerate(representatives.iterrows()):
            order = np.argsort(-np.abs(contributions[i]))[:5]
            for rank, feature_idx in enumerate(order, 1):
                contribution = float(contributions[i, feature_idx])
                local_rows.append({
                    "country_code": country, "shop_id": row.shop_id, "item_id": row.item_id,
                    "product_name": row.product_name, "model": best_name,
                    "predicted_log_monthly_sold_proxy": float(predictions_rep[i]),
                    "driver_rank": rank, "feature": design.feature_names[feature_idx],
                    "feature_label": _friendly_feature(design.feature_names[feature_idx]),
                    "direction": "positive" if contribution >= 0 else "negative",
                    "contribution_log_units": contribution,
                    "business_explanation": (
                        f"{_friendly_feature(design.feature_names[feature_idx]).capitalize()} "
                        f"{'raises' if contribution >= 0 else 'lowers'} the explanatory benchmark estimate."
                    ),
                })
    return metrics, predictions, pd.DataFrame(importance_rows), pd.DataFrame(local_rows)
