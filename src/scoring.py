from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


SCORE_COMPONENTS = [
    "engagement_strength", "sold_pct_peer", "price_competitiveness",
    "promotion_efficiency", "shop_credibility", "conversion_gap",
]

WEIGHT_CONFIGS: dict[str, dict[str, float]] = {
    "balanced": {
        "engagement_strength": 0.25, "sold_pct_peer": 0.20, "price_competitiveness": 0.10,
        "promotion_efficiency": 0.15, "shop_credibility": 0.10, "conversion_gap": 0.20,
    },
    "growth_opportunity": {
        "engagement_strength": 0.25, "sold_pct_peer": 0.10, "price_competitiveness": 0.10,
        "promotion_efficiency": 0.10, "shop_credibility": 0.10, "conversion_gap": 0.35,
    },
    "hero_protection": {
        "engagement_strength": 0.25, "sold_pct_peer": 0.35, "price_competitiveness": 0.05,
        "promotion_efficiency": 0.10, "shop_credibility": 0.15, "conversion_gap": 0.10,
    },
}


def calculate_score(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Score weights must sum to 1")
    score = sum(frame[name].fillna(0.5).clip(0, 1) * weight for name, weight in weights.items())
    return (100 * score).clip(0, 100)


def apply_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    sensitivity_rows: list[dict[str, float | str]] = []
    top_sets: dict[tuple[str, str], set[int]] = {}
    for config, weights in WEIGHT_CONFIGS.items():
        result[f"score_{config}"] = calculate_score(result, weights)
        result[f"rank_{config}"] = result.groupby("country_code")[f"score_{config}"].rank(ascending=False, method="min")
        for country, group in result.groupby("country_code"):
            top_sets[(config, country)] = set(group.nlargest(20, f"score_{config}").index)
    for country in sorted(result.country_code.unique()):
        base = top_sets[("balanced", country)]
        for config in WEIGHT_CONFIGS:
            other = top_sets[(config, country)]
            union = base | other
            jaccard = len(base & other) / len(union) if union else 1
            rank_corr = result.loc[result.country_code == country, ["rank_balanced", f"rank_{config}"]].corr(
                method="spearman"
            ).iloc[0, 1]
            sensitivity_rows.append({
                "country_code": country, "configuration": config,
                "top20_jaccard_vs_balanced": jaccard, "rank_spearman_vs_balanced": rank_corr,
            })
    result["opportunity_score"] = result["score_balanced"].round(2)
    return result, pd.DataFrame(sensitivity_rows)
