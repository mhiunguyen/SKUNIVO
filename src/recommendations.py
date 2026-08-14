from __future__ import annotations

import pandas as pd


def _label(row: pd.Series) -> str:
    if row.engagement_strength >= 0.75 and row.sold_pct_peer >= 0.75:
        return "Protect Hero SKU"
    if row.conversion_gap_candidate == 1:
        return "Conversion Opportunity"
    if row.engagement_strength >= 0.65 and row.sold_pct_peer >= 0.45 and row.promotion_mechanism_count <= 1 and row.displayed_discount_pct < 25:
        return "Promotion Test Candidate"
    if row.deep_discount_flag == 1 and row.sold_pct_peer < 0.40:
        return "Discount Efficiency Review"
    if row.engagement_strength < 0.30 and row.sold_pct_peer < 0.30:
        return "Low Priority"
    return "Maintain and Monitor"


def _reasons(row: pd.Series) -> list[str]:
    candidates: list[tuple[float, str]] = [
        (abs(row.likes_pct_peer - 0.5), f"Likes are at the {row.likes_pct_peer:.0%} peer percentile."),
        (abs(row.sold_pct_peer - 0.5), f"Historical sold-value proxy is at the {row.sold_pct_peer:.0%} peer percentile."),
        (abs(row.price_pct_country_category - 0.5), f"Price is at the {row.price_pct_country_category:.0%} country-category percentile."),
        (abs(row.discount_pct_peer - 0.5), f"Displayed discount is at the {row.discount_pct_peer:.0%} peer percentile."),
        (row.conversion_gap, f"Engagement exceeds sold-value strength by {row.conversion_gap:.0%}."),
        (0.25 if bool(row.is_official_shop) else 0, "The listing belongs to an official shop."),
        (abs(row.shop_credibility - 0.5), f"Shop credibility score is {row.shop_credibility:.0%}."),
    ]
    return [text for _, text in sorted(candidates, reverse=True)[:3]]


def assign_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["recommendation_label"] = result.apply(_label, axis=1)
    reasons = result.apply(_reasons, axis=1)
    result["reason_1"] = reasons.map(lambda x: x[0])
    result["reason_2"] = reasons.map(lambda x: x[1])
    result["reason_3"] = reasons.map(lambda x: x[2])
    result["country"] = result["country_code"]
    result["category"] = result["platform_category"]
    result["current_price"] = result["price"]
    result["discount_percent"] = result["displayed_discount_pct"]
    result["likes"] = result["liked_count"]
    result["monthly_sold_value_proxy"] = result["monthly_sold_value"]
    result["top_three_explanation_reasons"] = (
        result["reason_1"] + " | " + result["reason_2"] + " | " + result["reason_3"]
    )
    completeness = result[
        ["price", "liked_count", "rating_count", "history_sold_value", "platform_category", "rating_star"]
    ].notna().mean(axis=1)
    margin = (result["opportunity_score"] - 50).abs() / 50
    result["confidence_level"] = pd.cut(
        0.65 * completeness + 0.35 * margin,
        [-0.01, 0.62, 0.80, 1.01], labels=["Low", "Medium", "High"],
    ).astype(str)
    result["peer_group_benchmark"] = result.apply(
        lambda r: (
            f"{r.country_code} | {r.platform_category}; price pctl {r.price_pct_country_category:.0%}; "
            f"likes pctl {r.likes_pct_peer:.0%}; sold-value pctl {r.sold_pct_peer:.0%}"
        ), axis=1,
    )
    return result


OUTPUT_COLUMNS = [
    "country", "country_code", "shop_id", "shop_name", "item_id", "product_name", "category",
    "platform_category", "shop_category", "current_price", "price", "discount_percent",
    "displayed_discount_pct", "likes", "liked_count", "rating_count",
    "monthly_sold_value_proxy", "monthly_sold_value", "history_sold_value",
    "opportunity_score", "recommendation_label", "confidence_level",
    "top_three_explanation_reasons", "reason_1", "reason_2", "reason_3", "peer_group_benchmark",
]
