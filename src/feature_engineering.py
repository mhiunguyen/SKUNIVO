from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(default, index=n.index, dtype=float)
    valid = d.notna() & n.notna() & d.ne(0)
    result.loc[valid] = n.loc[valid] / d.loc[valid]
    return result


def group_percentile(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.Series:
    numeric = pd.to_numeric(frame[value], errors="coerce")
    return numeric.groupby([frame[g] for g in groups], dropna=False).rank(pct=True, method="average").fillna(0.5)


def engineer_features(latest: pd.DataFrame, rare_brand_min: int = 10) -> pd.DataFrame:
    frame = latest.copy()
    numeric = [
        "price", "price_original", "price_before_promo", "discount_percent", "promotion_id",
        "voucher_discount", "voucher_min_spend", "history_sold_value", "monthly_sold_value",
        "liked_count", "rating", "rating_count", "follower_count", "rating_star",
        "response_rate", "cancellation_rate",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["displayed_discount_pct"] = frame["discount_percent"].fillna(0).clip(0, 100)
    frame["price_ratio_original"] = safe_divide(frame["price"], frame["price_original"], 1.0).clip(0, 5)
    frame["price_ratio_pre_promo"] = safe_divide(frame["price"], frame["price_before_promo"], 1.0).clip(0, 5)
    peer = ["country_code", "platform_category"]
    frame["price_pct_country_category"] = group_percentile(frame, "price", peer)
    frame["price_pct_shop"] = group_percentile(frame, "price", ["country_code", "shop_id"])
    frame["price_band"] = pd.cut(
        frame["price_pct_country_category"], [-0.01, 0.33, 0.67, 1.01],
        labels=["Entry-level", "Mid-market", "Premium"],
    ).astype(str)
    frame["has_displayed_discount"] = frame["displayed_discount_pct"].gt(0).astype(int)
    frame["has_voucher"] = (
        frame["voucher_discount"].fillna(0).gt(0) | frame["voucher_code"].notna()
    ).astype(int)
    frame["has_promotion_id"] = frame["promotion_id"].fillna(0).gt(0).astype(int)
    frame["is_promoted"] = frame[["has_displayed_discount", "has_voucher", "has_promotion_id"]].max(axis=1)
    frame["promotion_mechanism_count"] = frame[
        ["has_displayed_discount", "has_voucher", "has_promotion_id"]
    ].sum(axis=1)
    frame["deep_discount_flag"] = frame["displayed_discount_pct"].ge(50).astype(int)
    frame["discount_pct_peer"] = group_percentile(frame, "displayed_discount_pct", peer)
    frame["log1p_liked_count"] = np.log1p(frame["liked_count"].fillna(0).clip(lower=0))
    frame["log1p_rating_count"] = np.log1p(frame["rating_count"].fillna(0).clip(lower=0))
    frame["likes_to_rating_ratio"] = safe_divide(frame["liked_count"], frame["rating_count"], 0).clip(0, 1_000)
    frame["likes_pct_peer"] = group_percentile(frame, "liked_count", peer)
    frame["rating_count_pct_peer"] = group_percentile(frame, "rating_count", peer)
    frame["sold_pct_peer"] = group_percentile(frame, "history_sold_value", peer)
    frame["high_like_indicator"] = frame["likes_pct_peer"].ge(0.75).astype(int)
    frame["shop_assortment_size"] = frame.groupby(["country_code", "shop_id"])["item_id"].transform("nunique")
    frame["follower_pct_country"] = group_percentile(frame, "follower_count", ["country_code"])
    frame["shop_rating_pct_country"] = group_percentile(frame, "rating_star", ["country_code"])
    brand = frame["brand"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    counts = brand.groupby(frame["country_code"]).transform("count")
    pair_counts = frame.assign(_brand=brand).groupby(["country_code", "_brand"])["item_id"].transform("count")
    frame["brand_grouped"] = brand.where(pair_counts >= rare_brand_min, "Other")
    frame["country_category"] = frame["country_code"].astype(str) + " | " + frame["platform_category"].fillna("Unmapped").astype(str)
    frame["engagement_strength"] = frame[["likes_pct_peer", "rating_count_pct_peer"]].mean(axis=1)
    frame["price_competitiveness"] = 1 - frame["price_pct_country_category"]
    response = frame["response_rate"].fillna(frame.groupby("country_code")["response_rate"].transform("median")).fillna(0) / 100
    cancellation = 1 - frame["cancellation_rate"].fillna(0).clip(0, 100) / 100
    official = frame["is_official_shop"].fillna(False).astype(int)
    frame["shop_credibility"] = pd.concat([
        frame["follower_pct_country"], frame["shop_rating_pct_country"], response, cancellation, official
    ], axis=1).mean(axis=1).clip(0, 1)
    frame["promotion_efficiency"] = (
        0.55 * frame["sold_pct_peer"] + 0.45 * frame["engagement_strength"]
    ) * (1 - 0.35 * frame["discount_pct_peer"])
    frame["conversion_gap"] = (frame["engagement_strength"] - frame["sold_pct_peer"]).clip(lower=0)
    frame["conversion_gap_candidate"] = (
        frame["high_like_indicator"].eq(1) & frame["conversion_gap"].ge(0.20)
    ).astype(int)
    frame["log1p_history_sold"] = np.log1p(frame["history_sold_value"].fillna(0).clip(lower=0))
    frame["log1p_price"] = np.log1p(frame["price"].fillna(0).clip(lower=0))
    frame["log1p_follower_count"] = np.log1p(frame["follower_count"].fillna(0).clip(lower=0))
    frame["voucher_discount_missing"] = frame["voucher_discount"].isna().astype(int)
    frame["voucher_min_spend_missing"] = frame["voucher_min_spend"].isna().astype(int)
    return frame


ACTIONABLE_NUMERIC = [
    "log1p_price", "price_ratio_original", "price_ratio_pre_promo",
    "price_pct_country_category", "price_pct_shop", "displayed_discount_pct",
    "discount_pct_peer", "has_displayed_discount", "has_voucher", "has_promotion_id",
    "is_promoted", "promotion_mechanism_count", "deep_discount_flag",
    "voucher_discount", "voucher_min_spend", "voucher_discount_missing",
    "voucher_min_spend_missing", "log1p_liked_count", "rating", "log1p_rating_count",
    "likes_to_rating_ratio", "high_like_indicator", "log1p_follower_count",
    "rating_star", "is_official_shop", "response_rate", "cancellation_rate",
    "shop_assortment_size",
]
CATEGORICAL = ["platform_category", "shop_category", "brand_grouped", "price_band"]


@dataclass
class DesignMatrix:
    x: np.ndarray
    feature_names: list[str]
    numeric_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    category_levels: dict[str, list[str]]


def fit_design_matrix(frame: pd.DataFrame, descriptive: bool = False) -> DesignMatrix:
    numeric_names = ACTIONABLE_NUMERIC + (["log1p_history_sold", "sold_pct_peer"] if descriptive else [])
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    medians = np.array([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
        for column in numeric.T
    ])
    numeric = np.where(np.isfinite(numeric), numeric, medians)
    means = numeric.mean(axis=0)
    scales = numeric.std(axis=0)
    scales = np.where(scales > 1e-9, scales, 1)
    z = (numeric - means) / scales
    matrices = [z]
    names = list(numeric_names)
    levels: dict[str, list[str]] = {}
    for column in CATEGORICAL:
        values = frame[column].fillna("Unknown").astype(str)
        cats = sorted(values.unique().tolist())
        levels[column] = cats
        if len(cats) > 1:
            cats = cats[1:]
            matrices.append(np.column_stack([(values == cat).astype(float) for cat in cats]))
            names.extend([f"{column}={cat}" for cat in cats])
    return DesignMatrix(np.column_stack(matrices), names, numeric_names, medians, means, scales, levels)


def transform_design_matrix(frame: pd.DataFrame, design: DesignMatrix) -> np.ndarray:
    numeric = frame[design.numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    numeric = np.where(np.isfinite(numeric), numeric, design.medians)
    matrices = [(numeric - design.means) / design.scales]
    for column in CATEGORICAL:
        values = frame[column].fillna("Unknown").astype(str)
        cats = design.category_levels[column][1:]
        if cats:
            matrices.append(np.column_stack([(values == cat).astype(float) for cat in cats]))
    return np.column_stack(matrices)
