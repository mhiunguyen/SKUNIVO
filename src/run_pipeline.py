from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL

from .charts import contact_sheet, create_required_charts
from .data_pipeline import build_latest_product_table
from .feature_engineering import ACTIONABLE_NUMERIC, engineer_features
from .modeling import run_model_experiments
from .recommendations import OUTPUT_COLUMNS, assign_recommendations
from .scoring import WEIGHT_CONFIGS, apply_scores

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"
LOGGER = logging.getLogger("decision_copilot")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |")
    return "\n".join(lines)


def target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in frame.groupby("country_code"):
        target = pd.to_numeric(group["monthly_sold_value"], errors="coerce")
        valid = target.dropna()
        rows.append({
            "country_code": country, "rows": len(group), "target_non_missing": len(valid),
            "target_missing": int(target.isna().sum()), "missing_pct": target.isna().mean(),
            "zero_share": float(valid.eq(0).mean()) if len(valid) else np.nan,
            "min": valid.min(), "q25": valid.quantile(.25), "median": valid.median(),
            "q75": valid.quantile(.75), "max": valid.max(),
        })
    return pd.DataFrame(rows)


def missingness(frame: pd.DataFrame) -> pd.DataFrame:
    selected = sorted(set(ACTIONABLE_NUMERIC + [
        "platform_category", "shop_category", "brand_grouped", "monthly_sold_value",
        "history_sold_value", "opportunity_score",
    ]))
    rows = []
    for country, group in frame.groupby("country_code"):
        for column in selected:
            rows.append({
                "country_code": country, "field": column, "missing_count": int(group[column].isna().sum()),
                "missing_pct": float(group[column].isna().mean()),
            })
    return pd.DataFrame(rows)


def ml_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = metrics.loc[metrics.validation_scheme == "group_5_fold"].copy()
    baseline = group.loc[group.model == "DummyMedian", [
        "country_code", "experiment", "rmse_log", "ndcg_at_20", "top_decile_lift"
    ]].rename(
        columns={
            "rmse_log": "baseline_rmse", "ndcg_at_20": "baseline_ndcg",
            "top_decile_lift": "baseline_top_decile_lift",
        }
    )
    candidates = group.loc[group.model != "DummyMedian"].sort_values(
        ["country_code", "experiment", "rmse_log", "ndcg_at_20"], ascending=[True, True, True, False]
    ).groupby(["country_code", "experiment"], as_index=False).first()
    comparison = candidates.merge(baseline, on=["country_code", "experiment"], how="left")
    comparison["rmse_improvement_pct"] = (comparison.baseline_rmse - comparison.rmse_log) / comparison.baseline_rmse
    comparison["ndcg_change"] = comparison.ndcg_at_20 - comparison.baseline_ndcg
    comparison["materially_beats_baseline"] = (
        comparison.rmse_improvement_pct.ge(.05)
        & comparison.ndcg_change.ge(.02)
        & comparison.top_decile_lift.ge(1.10)
    )
    return group, comparison


def write_documents(
    audit: dict[str, Any], scored: pd.DataFrame, metrics: pd.DataFrame,
    importance: pd.DataFrame, local: pd.DataFrame, sensitivity: pd.DataFrame,
    comparison: pd.DataFrame, chart_paths: list[Path],
) -> None:
    top = scored.sort_values(["country_code", "opportunity_score"], ascending=[True, False]).groupby(
        "country_code", as_index=False
    ).head(5)
    metric_display = metrics.copy()
    for column in ["mae_log", "rmse_log", "spearman", "top_decile_lift", "ndcg_at_20"]:
        metric_display[column] = metric_display[column].round(4)
    comp_display = comparison[[
        "country_code", "experiment", "model", "rmse_log", "baseline_rmse",
        "rmse_improvement_pct", "ndcg_at_20", "baseline_ndcg", "top_decile_lift",
        "baseline_top_decile_lift", "materially_beats_baseline",
    ]].copy()
    comp_display["rmse_improvement_pct"] = (100 * comp_display["rmse_improvement_pct"]).round(1).astype(str) + "%"
    technical = [
        "# Technical Summary — AI Decision Copilot",
        "",
        "## Scope",
        "",
        "Cross-sectional, explainable product prioritization only. The prototype does not forecast future transactional demand or estimate causal promotion lift.",
        "",
        "## Data audit",
        "",
        f"- Loaded five logical table families; latest product table: {audit['latest_rows']:,} rows.",
        f"- Snapshot dates: {', '.join(audit['snapshot_dates'])}.",
        f"- Exact product duplicates removed: {audit['exact_duplicates_removed']}; the known Vietnam shop 289646907 file contributes {audit['known_vn_shop_289646907_duplicates']}.",
        f"- Product → shop match rate: {audit['shop_join']['match_rate']:.2%}.",
        f"- Product → platform category match rate: {audit['platform_category_join']['match_rate']:.2%}.",
        f"- Latest product → aggregated shop-category match rate: {audit['latest_product_category_join']['match_rate']:.2%}.",
        "- The product-category table is aggregated to one primary shop category before joining, preventing many-to-many row multiplication.",
        "",
        "## Rule-based score",
        "",
        "Balanced weights:",
        "",
        markdown_table(pd.DataFrame([{"component": k, "weight": v} for k, v in WEIGHT_CONFIGS["balanced"].items()])),
        "",
        "All score components are 0–1 peer-relative signals within country and platform category; the final score is scaled to 0–100.",
        "",
        "## Model fallback",
        "",
        "CatBoost, LightGBM, scikit-learn, SHAP, and SciPy were unavailable. The documented fallback uses NumPy implementations of a median baseline, regularized linear regression, and deterministic gradient-boosted regression stumps.",
        "",
        "## Grouped validation results",
        "",
        markdown_table(metric_display),
        "",
        "## Best non-trivial model versus baseline",
        "",
        markdown_table(comp_display),
        "",
        "The descriptive experiment includes historical sold value and is explicitly leakage-prone for future prediction. It is retained only as a cross-sectional explanatory benchmark.",
        "",
        "## Explainability",
        "",
        f"Global importance contains {len(importance)} feature rows. Local additive explanations were generated for {local[['country_code','shop_id','item_id']].drop_duplicates().shape[0]} representative products.",
        "",
        "## Ranking robustness",
        "",
        markdown_table(sensitivity.round(4)),
        "",
        "## Reproducibility",
        "",
        "From `D:\\YOUNGHTT` in PowerShell:",
        "",
        "```powershell",
        "& 'C:\\Users\\mhiuq\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -m src.run_pipeline",
        "```",
    ]
    (OUTPUTS / "technical_summary.md").write_text("\n".join(technical), encoding="utf-8")

    examples = top[[
        "country_code", "shop_name", "product_name", "opportunity_score",
        "recommendation_label", "reason_1", "reason_2", "reason_3",
    ]].copy()
    proposal = [
        "# Proposal Insights — AI Decision Copilot",
        "",
        "## Business problem",
        "",
        "Merchandising teams must decide which listings to protect, test, review, or deprioritize across large assortments with uneven engagement, pricing, promotions, and shop credibility.",
        "",
        "## Why current decisions are difficult",
        "",
        "Marketplace signals are fragmented, cumulative, highly skewed, and strongly influenced by country, category, shop, and listing maturity. Deep discounts and high star ratings alone do not reliably identify opportunity.",
        "",
        "## Dataset capabilities",
        "",
        "The data supports cross-sectional comparison using product, shop, category, local price, displayed promotion, ratings, likes, and source-reported sold-value proxies.",
        "",
        "## Dataset limitations",
        "",
        "There are only three snapshot dates and no orders, customers, costs, campaign spend, inventory history, realized revenue, exposure/control assignment, or currency metadata.",
        "",
        "## Proposed AI solution",
        "",
        "An explainable product opportunity and promotion prioritization copilot that combines transparent peer-normalized scoring with an ML-assisted explanatory benchmark.",
        "",
        "## Decision workflow",
        "",
        "1. Refresh and validate latest listings. 2. Benchmark within country/category peers. 3. Assign a 0–100 opportunity score. 4. Produce decision-support labels and reasons. 5. Review top candidates with a merchandiser. 6. Route promotion candidates into controlled tests.",
        "",
        "## Model approach",
        "",
        "Track A is a transparent weighted score with balanced, growth-opportunity, and hero-protection configurations. Track B estimates log1p monthly sold-value proxy using shop-grouped cross-validation. The actionable experiment excludes historical sold value; the descriptive experiment includes it and is labeled leakage-prone.",
        "",
        "## Explainability approach",
        "",
        "Every recommendation shows peer percentiles and three business-language reasons. Global importance summarizes the best actionable models; local additive contributions explain representative products without presenting raw model coefficients as decisions.",
        "",
        "## Example recommendations",
        "",
        markdown_table(examples),
        "",
        "## Evaluation strategy",
        "",
        "Use grouped five-fold validation and leave-one-shop-out evaluation. Report log-scale MAE/RMSE, Spearman ranking correlation, top-decile lift, and NDCG@20. Monitor top-20 score overlap across weight configurations.",
        "",
        "## Business value",
        "",
        "The copilot creates a consistent review queue, exposes conversion-gap candidates, distinguishes hero protection from discount-efficiency review, and makes prioritization auditable.",
        "",
        "## Feasibility and scalability",
        "",
        "The prototype runs with pandas, NumPy, and Pillow only. Feature computation and scoring are deterministic, country-aware, and modular; stronger libraries can replace the fallback models without changing the decision interface.",
        "",
        "## Future data roadmap",
        "",
        "Add listing-age history, orders and realized prices, promotion exposure/control, stock history, costs, campaign metrics, refunds, and currency definitions. These additions would enable causal lift measurement, forecasting, margin-aware decisions, and inventory optimization.",
        "",
        "## Honest boundary",
        "",
        "“This prototype prioritizes products using observed marketplace signals. It does not estimate causal promotion lift or forecast transactional demand with the current three-day snapshot dataset.”",
        "",
        "## Proposal-ready charts",
        "",
    ]
    proposal += [f"- `{path.relative_to(ROOT).as_posix()}`" for path in chart_paths]
    (OUTPUTS / "proposal_insights.md").write_text("\n".join(proposal), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    latest, audit, _ = build_latest_product_table(ROOT)
    features = engineer_features(latest)
    scored, sensitivity = apply_scores(features)
    scored = assign_recommendations(scored)
    if scored.empty or scored["opportunity_score"].isna().any():
        raise RuntimeError("Scoring produced an empty or invalid output")

    scored.to_csv(OUTPUTS / "processed_latest_products.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(OUTPUTS / "model_dataset.csv", index=False, encoding="utf-8-sig")
    recommendations = scored[OUTPUT_COLUMNS].sort_values(
        ["country_code", "opportunity_score"], ascending=[True, False]
    )
    recommendations.to_csv(OUTPUTS / "product_recommendations.csv", index=False, encoding="utf-8-sig")
    recommendations.groupby("country_code", as_index=False).head(10).to_csv(
        OUTPUTS / "top_opportunities_by_country.csv", index=False, encoding="utf-8-sig"
    )

    metrics, predictions, importance, local = run_model_experiments(scored)
    metrics.to_csv(OUTPUTS / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUTS / "model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUTPUTS / "feature_importance.csv", index=False, encoding="utf-8-sig")
    local.to_csv(OUTPUTS / "local_explanations.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUTS / "score_sensitivity.csv", index=False, encoding="utf-8-sig")
    missingness(scored).to_csv(OUTPUTS / "modeling_field_missingness.csv", index=False, encoding="utf-8-sig")
    target_distribution(scored).to_csv(OUTPUTS / "target_distribution.csv", index=False, encoding="utf-8-sig")
    (OUTPUTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    versions = {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "Pillow": PIL.__version__, "random_seed": SEED,
        "unavailable_requested_libraries": ["catboost", "lightgbm", "sklearn", "shap", "scipy"],
    }
    (OUTPUTS / "package_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")

    _, comparison = ml_comparison(metrics)
    chart_paths = create_required_charts(scored, metrics, importance, sensitivity, CHARTS)
    sheet = contact_sheet(chart_paths, CHARTS)
    write_documents(audit, scored, metrics, importance, local, sensitivity, comparison, chart_paths)

    return {
        "latest_products": len(scored), "countries": scored.country_code.value_counts().to_dict(),
        "recommendations": scored.recommendation_label.value_counts().to_dict(),
        "charts": len(chart_paths), "local_explained_products": local[["country_code","shop_id","item_id"]].drop_duplicates().shape[0],
        "model_comparison": comparison.to_dict("records"),
        "contact_sheet": str(sheet.relative_to(ROOT)),
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), ensure_ascii=False, indent=2, default=str))
