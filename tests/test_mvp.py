from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app_components.ai_benchmark import ai_decision_brief
from app_components.data_loader import load_app_data, validate_assets
from app_components.decision_log import append_decision, make_decision_row
from app_components.feedback import append_feedback, is_public_mode, make_feedback_row
from app_components.filters import apply_product_filters, format_local_price
from app_components.recommendation_ui import (
    SCORE_PRESETS,
    calculate_what_if_score,
)
from app_components.persistence import deliver_record


class ScoreFormulaTests(unittest.TestCase):
    def test_required_formula_cases(self) -> None:
        keys = list(SCORE_PRESETS["Balanced product"])
        cases = [
            ({key: 0 for key in keys}, 0),
            ({key: 100 for key in keys}, 100),
            ({key: (100 if key == "engagement_strength" else 0) for key in keys}, 25),
            ({key: (100 if key == "sold_value_strength" else 0) for key in keys}, 20),
            (
                {key: (100 if key == "conversion_gap_opportunity" else 0) for key in keys},
                20,
            ),
            ({key: 50 for key in keys}, 50),
        ]
        for components, expected in cases:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(calculate_what_if_score(components), expected)

    def test_score_is_clamped(self) -> None:
        self.assertEqual(calculate_what_if_score({"engagement_strength": 1000}), 25)
        self.assertEqual(
            calculate_what_if_score({key: -100 for key in SCORE_PRESETS["Balanced product"]}),
            0,
        )


class DataAndFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_app_data()
        cls.products = cls.data["products"]

    def test_required_assets_and_row_count(self) -> None:
        self.assertEqual(validate_assets(), [])
        self.assertEqual(len(self.products), 1157)
        self.assertEqual(self.products["shop_id"].nunique(), 20)
        self.assertEqual(self.products["country_code"].nunique(), 2)
        self.assertEqual(self.products["recommendation_label"].nunique(), 6)

    def test_product_keys_are_unique(self) -> None:
        self.assertFalse(
            self.products.duplicated(["country_code", "shop_id", "item_id"]).any()
        )

    def test_country_and_empty_filter_results(self) -> None:
        indonesia = apply_product_filters(self.products, {"countries": ["id"]})
        self.assertGreater(len(indonesia), 0)
        self.assertEqual(set(indonesia["country_code"]), {"id"})
        empty = apply_product_filters(
            self.products,
            {"countries": ["id"], "score_range": (99.9, 100.0)},
        )
        self.assertTrue(empty.empty)

    def test_market_price_labels(self) -> None:
        self.assertTrue(format_local_price(1000, "id").startswith("IDR "))
        self.assertTrue(format_local_price(1000, "vn").startswith("VND "))
        self.assertNotIn("USD", format_local_price(1000, "id"))

    def test_optional_values_can_be_missing(self) -> None:
        frame = self.products.head(1).copy()
        frame["shop_category"] = pd.NA
        result = apply_product_filters(frame, {"countries": ["id", "vn"]})
        self.assertEqual(len(result), 1)

    def test_ai_benchmark_is_visible_and_honest(self) -> None:
        modeled = self.products[self.products["ai_contextual_sold_benchmark"].notna()]
        self.assertEqual(len(modeled), 1157)
        self.assertTrue(
            set(modeled["ai_model_confidence"].unique()).issubset({"High", "Low"})
        )
        brief = ai_decision_brief(modeled.iloc[0])
        self.assertIn("benchmark", brief.lower())
        self.assertIn("not a future-sales forecast", brief.lower())


class FeedbackTests(unittest.TestCase):
    def test_public_mode_uses_request_url(self) -> None:
        self.assertTrue(is_public_mode("https://skunivo.streamlit.app/"))
        self.assertFalse(is_public_mode("http://localhost:8501/"))

    def test_feedback_appends_without_overwrite(self) -> None:
        values = {
            "participant_role": "Analyst",
            "test_scenario": "End-to-end guided test",
            "usefulness_rating": 4,
            "explanation_clarity_rating": 4,
            "trust_rating": 3,
            "navigation_rating": 5,
            "would_use": "Yes",
            "most_useful_feature": "Explanation",
            "confusing_element": "None",
            "improvement_suggestion": "More history",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.csv"
            append_feedback(make_feedback_row(values), path)
            append_feedback(make_feedback_row(values), path)
            frame = pd.read_csv(path)
            self.assertEqual(len(frame), 2)


class PersistenceAndDecisionTests(unittest.TestCase):
    def test_decision_appends_without_overwrite(self) -> None:
        product = {
            "country_code": "id",
            "shop_id": 1,
            "shop_name": "Shop",
            "item_id": 2,
            "product_name": "Product",
            "platform_category": "Beauty",
            "opportunity_score": 70,
            "recommendation_label": "Protect Hero SKU",
            "confidence_level": "High",
            "ai_benchmark_signal": "Above contextual benchmark",
            "ai_model_confidence": "High",
        }
        values = {
            "reviewer_role": "Category manager",
            "decision_status": "Accept recommendation",
            "selected_action": "Protect current execution",
            "decision_rationale": "Strong peer evidence",
            "success_metric": "Peer position",
            "review_date": "2026-08-12",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            append_decision(make_decision_row(product, values), path)
            append_decision(make_decision_row(product, values), path)
            self.assertEqual(len(pd.read_csv(path)), 2)

    def test_google_sheets_delivery_success(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return b'{"ok": true}'

        with patch("app_components.persistence.urlopen", return_value=FakeResponse()):
            result = deliver_record(
                "feedback",
                {"participant_role": "Judge or mentor"},
                webhook_url="https://example.com/exec",
                webhook_token="secret",
            )
        self.assertTrue(result.delivered)
        self.assertTrue(result.configured)


if __name__ == "__main__":
    unittest.main()
