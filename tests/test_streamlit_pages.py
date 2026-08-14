from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


PAGES = [
    "Home",
    "Executive Overview",
    "Product Prioritization",
    "Product Explanation",
    "Decision Log",
    "What-if Score Explorer",
    "Methodology and Transparency",
    "User Feedback",
]


class StreamlitPageSmokeTests(unittest.TestCase):
    @staticmethod
    def _open_page(app: AppTest, page: str) -> AppTest:
        app.radio(key="nav_radio").set_value(page).run()
        return app

    def test_every_page_opens_without_exception(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        navigation = app.radio(key="nav_radio")
        self.assertEqual(navigation.value, "Home")
        for page in PAGES[1:]:
            with self.subTest(page=page):
                self._open_page(app, page)
                self.assertEqual(list(app.exception), [])
                self.assertEqual(app.radio(key="nav_radio").value, page)

    def test_country_specific_price_filter_and_download(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "Product Prioritization")
        market = next(
            control
            for control in app.get("button_group")
            if control.label == "Decision market"
        )
        self.assertEqual(market.value, "id")
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.slider(key="filter_price_id").label, "Current price · local IDR units")
        self.assertGreaterEqual(len(app.get("download_button")), 1)

    def test_judge_demo_starts_in_indonesia_queue(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        start = next(
            button for button in app.button if button.label == "Start 3-minute judge demo →"
        )
        start.click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.radio(key="nav_radio").value, "Product Prioritization")
        self.assertTrue(app.session_state["demo_guide_started"])
        self.assertEqual(app.session_state["filter_country"], "id")

    def test_executive_country_views_render_without_altair_schema_errors(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "Executive Overview")
        market_view = next(
            control
            for control in app.get("button_group")
            if control.label == "Market view"
        )
        for market in ("Indonesia", "Vietnam"):
            with self.subTest(market=market):
                market_view.set_value(market).run()
                self.assertEqual(list(app.exception), [])
                market_view = next(
                    control
                    for control in app.get("button_group")
                    if control.label == "Market view"
                )

    def test_open_product_explanation_button_navigates(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "Product Prioritization")
        open_button = next(
            button for button in app.button if button.label == "Open product explanation →"
        )
        open_button.click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.radio(key="nav_radio").value, "Product Explanation")
        self.assertIn("selected_product_key", app.session_state)

    def test_presets_update_score_components(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "What-if Score Explorer")
        growth_button = next(button for button in app.button if button.label == "Growth opportunity")
        growth_button.click().run()
        self.assertEqual(app.slider(key="whatif_engagement_strength").value, 82)
        self.assertEqual(app.slider(key="whatif_conversion_gap_opportunity").value, 90)
        self.assertEqual(list(app.exception), [])

    def test_ai_benchmark_and_decision_log_are_visible(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "Product Explanation")
        self.assertTrue(
            any(
                expander.label == "Supporting evidence and model details"
                for expander in app.expander
            )
        )
        self.assertTrue(
            any("AI-assisted contextual benchmark" in str(markdown.value) for markdown in app.markdown)
        )
        self._open_page(app, "Decision Log")
        save = next(button for button in app.button if button.label == "Save decision")
        save.click().run()
        self.assertTrue(
            any("asterisk" in str(error.value).lower() for error in app.error),
            "Expected Decision Log required-field validation message.",
        )
        self.assertEqual(list(app.exception), [])

    def test_feedback_rejects_empty_required_fields(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self._open_page(app, "User Feedback")
        submit = next(button for button in app.button if button.label == "Submit prototype feedback")
        submit.click().run()
        self.assertTrue(
            any("asterisk" in str(error.value).lower() for error in app.error),
            "Expected required-field validation message.",
        )


if __name__ == "__main__":
    unittest.main()
