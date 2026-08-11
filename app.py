from __future__ import annotations

import base64
import html
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from app_components.ai_benchmark import ai_decision_brief
from app_components.charts import (
    contribution_frame,
    recommendation_mix,
    score_band_summary,
    show_image_chart,
)
from app_components.decision_log import (
    append_decision,
    decision_csv_bytes,
    make_decision_row,
    validate_decision,
)
from app_components.data_loader import (
    DataValidationError,
    ROOT,
    display_value,
    load_app_data,
)
from app_components.feedback import (
    append_feedback,
    feedback_csv_bytes,
    is_public_mode,
    make_feedback_row,
    validate_feedback,
)
from app_components.filters import (
    ACTIVE_REVIEW_LABELS,
    SORT_OPTIONS,
    active_review_count,
    apply_product_filters,
    format_local_price,
    safe_range,
    sort_products,
)
from app_components.i18n import translate
from app_components.recommendation_ui import (
    COMPONENT_LABELS,
    GUIDANCE,
    SCORE_PRESETS,
    SCORE_WEIGHTS,
    calculate_what_if_score,
    illustrative_tier,
    score_contributions,
)
from app_components.persistence import deliver_record, google_sheets_config
from app_components.styles import APP_CSS


BRAND_DIR = ROOT / "assets" / "branding"
LOGO_ON_DARK = BRAND_DIR / "skunivo-logo-final-on-dark.svg"
FAVICON_MARK = BRAND_DIR / "skunivo-favicon.svg"
FAVICON = BRAND_DIR / "skunivo-favicon-32.png"


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return f"data:{mime_type};base64,{encoded}"


st.set_page_config(
    page_title="SKUNIVO",
    page_icon=str(FAVICON),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "skunivo_app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

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


def t(text: str) -> str:
    return translate(text, st.session_state.get("language", "en"))


def confidence_text(value: str, *, model: bool = False) -> str:
    if st.session_state.get("language") == "vi":
        label = t("Model confidence") if model else t("Confidence")
        return f"{label}: {t(str(value))}"
    return f"{value} {'model confidence' if model else 'confidence'}"


def localized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate display labels without changing the underlying decision data."""
    display = frame.copy()
    for column in display.select_dtypes(include=["object", "string"]).columns:
        display[column] = display[column].map(
            lambda value: t(str(value)) if value is not None and not pd.isna(value) else value
        )
    display.columns = [t(str(column)) for column in display.columns]
    return display

BOUNDARY = (
    "This prototype prioritizes products using observed marketplace signals. "
    "It does not estimate causal promotion lift or forecast transactional demand "
    "with the current three-day snapshot dataset."
)

PAGE_SUBTITLES = {
    "Executive Overview": "A normalized portfolio view for fast merchandising triage.",
    "Product Prioritization": "Filter and rank listings within their market context.",
    "Product Explanation": "Audit the evidence behind one product recommendation.",
    "Decision Log": "Accept, override, or defer a recommendation and preserve the learning record.",
    "What-if Score Explorer": "Explore the transparent score mechanics—not future outcomes.",
    "Methodology and Transparency": "Understand the data, scoring logic, evaluation, and limits.",
    "User Feedback": "Help us evaluate usefulness, clarity, trust, and navigation.",
}


def navigate(page: str) -> None:
    st.session_state.active_page = page
    st.session_state.nav_radio = t(page)


def open_product_explanation(product_key: str) -> None:
    st.session_state.selected_product_key = product_key
    if st.session_state.get("demo_guide_started", False):
        st.session_state.demo_step_explanation = True
    navigate("Product Explanation")


def open_decision_log(product_key: str) -> None:
    st.session_state.selected_product_key = product_key
    navigate("Decision Log")


def sync_navigation() -> None:
    page_by_label = {t(page): page for page in PAGES}
    st.session_state.active_page = page_by_label.get(
        st.session_state.nav_radio,
        st.session_state.active_page,
    )


def page_header(title: str) -> None:
    st.markdown(
        f"""
        <div class="mp-page-head">
          <span class="mp-kicker">{html.escape(t("SKUNIVO · DECISION SUPPORT"))}</span>
          <h1>{html.escape(t(title))}</h1>
          <p>{html.escape(t(PAGE_SUBTITLES[title]))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, copy: str = "") -> None:
    st.markdown(f'<h2 class="mp-section-title">{html.escape(t(title))}</h2>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<p class="mp-section-copy">{html.escape(t(copy))}</p>', unsafe_allow_html=True)


def render_boundary() -> None:
    st.markdown(f'<div class="mp-boundary">{html.escape(t(BOUNDARY))}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="mp-footer">
          <strong>SKUNIVO</strong> · {html.escape(t("Built by Team YOUNGHTT"))}<br>
          {html.escape(t("Human review remains part of every decision. Scores are peer-relative and use precomputed marketplace signals."))}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value) -> str:
    if value is None or pd.isna(value):
        return t("Not available")
    return f"{float(value):.0%}"


def format_number(value, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return t("Not available")
    return f"{float(value):,.{digits}f}"


def bool_label(value) -> str:
    if value is None or pd.isna(value):
        return t("Not available")
    return t("Yes") if bool(value) else t("No")


def metric_cards(items: list[tuple[str, str]]) -> None:
    markup = "".join(
        f'<div class="mp-metric"><strong>{html.escape(value)}</strong><span>{html.escape(t(label))}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="mp-metric-grid">{markup}</div>', unsafe_allow_html=True)


DEMO_STEPS = [
    ("demo_step_prioritization", "Open the product priority queue"),
    ("demo_step_explanation", "Review an AI-assisted product explanation"),
    ("demo_step_decision", "Accept or override a recommendation"),
]


def mark_demo_step(key: str) -> None:
    if st.session_state.get("demo_guide_started", False):
        st.session_state[key] = True


def start_demo_guide() -> None:
    st.session_state.demo_guide_started = True
    st.session_state.demo_step_prioritization = True
    st.session_state.filter_country = "id"
    navigate("Product Prioritization")


def demo_guide_panel() -> None:
    with st.sidebar.expander(
        t("Demo Guide · 3 steps"),
        expanded=st.session_state.get("demo_guide_started", False),
    ):
        if not st.session_state.get("demo_guide_started", False):
            st.caption(t("Start from Home. Progress updates automatically in this browser session."))
        completed = sum(
            bool(st.session_state.get(key, False))
            for key, _ in DEMO_STEPS
        )
        st.progress(completed / len(DEMO_STEPS))
        for key, label in DEMO_STEPS:
            icon = "✓" if st.session_state.get(key, False) else "○"
            st.caption(f"{icon} {t(label)}")


def render_shell() -> None:
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"
    if "language" not in st.session_state:
        st.session_state.language = "en"

    with st.sidebar:
        sidebar_logo = image_data_uri(LOGO_ON_DARK)
        st.markdown(
            f"""
            <div class="mp-brand-lockup">
              <img class="mp-brand-wordmark" src="{sidebar_logo}" alt="SKUNIVO">
            </div>
            <div class="mp-brand-sub">{html.escape(t("Explainable e-commerce decision intelligence"))}</div>
            """,
            unsafe_allow_html=True,
        )
        st.segmented_control(
            t("Language"),
            ["en", "vi"],
            key="language",
            format_func=lambda value: "English" if value == "en" else "Tiếng Việt",
            selection_mode="single",
        )
        translated_pages = [t(page) for page in PAGES]
        expected_nav_label = t(st.session_state.active_page)
        if st.session_state.get("nav_radio") not in translated_pages:
            st.session_state.nav_radio = expected_nav_label
        st.radio(
            t("Navigate"),
            translated_pages,
            key="nav_radio",
            on_change=sync_navigation,
            label_visibility="collapsed",
        )
        st.divider()
        demo_guide_panel()
        st.caption(t("Decision-support prototype · Precomputed outputs only"))


def home_page(products: pd.DataFrame) -> None:
    top_product = products.sort_values("opportunity_score", ascending=False).iloc[0]
    hero_logo = image_data_uri(FAVICON_MARK)
    st.markdown(
        f"""
        <div class="mp-hero">
          <img class="mp-hero-mark" src="{hero_logo}" alt="" aria-hidden="true">
          <div class="mp-eyebrow">{html.escape(t("AI DECISION COPILOT FOR E-COMMERCE"))}</div>
          <h1>{html.escape(t("Turn marketplace signals into explainable product decisions."))}</h1>
          <p>{html.escape(t("SKUNIVO benchmarks each product against comparable listings in its local market, assigns a transparent opportunity score, and explains which products should be protected, tested, reviewed, maintained, or deprioritized."))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta_left, cta_middle, cta_right, _ = st.columns([1.5, 1.2, 1, 2.3])
    with cta_left:
        st.button(
            t("Start 3-minute judge demo →"),
            type="primary",
            width="stretch",
            on_click=start_demo_guide,
        )
    with cta_middle:
        st.button(
            t("Explore independently"),
            width="stretch",
            on_click=navigate,
            args=("Product Prioritization",),
        )
    with cta_right:
        st.button(
            t("View Methodology"),
            width="stretch",
            on_click=navigate,
            args=("Methodology and Transparency",),
        )

    metric_cards(
        [
            ("Latest product listings", f"{len(products):,}"),
            ("Shops", f"{products['shop_id'].nunique():,}"),
            ("Markets", f"{products['country_code'].nunique():,}"),
            ("Recommendation types", f"{products['recommendation_label'].nunique():,}"),
        ]
    )
    st.markdown(
        f"""
        <div class="mp-workflow">
          <div class="mp-step">{html.escape(t("Benchmark"))}</div><div class="mp-step">{html.escape(t("Score"))}</div>
          <div class="mp-step">{html.escape(t("Explain"))}</div><div class="mp-step">{html.escape(t("Review"))}</div>
          <div class="mp-step">{html.escape(t("Test"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header(
        "A recommendation you can audit",
        "One real listing from the current ranked output—shown without changing its score, label, confidence, or reasons.",
    )
    left, right = st.columns([1.25, 1])
    with left:
        country_class = "mp-market-id" if top_product["country_code"] == "id" else "mp-market-vn"
        reasons = "".join(
            f'<div class="mp-reason">{html.escape(t(str(top_product[name])))}</div>'
            for name in ("reason_1", "reason_2", "reason_3")
        )
        st.markdown(
            f"""
            <div class="mp-preview">
              <span class="mp-kicker">{html.escape(t("LIVE RECOMMENDATION PREVIEW"))}</span>
              <h3>{html.escape(str(top_product["product_name"]))}</h3>
              <p><span class="{country_class}">{html.escape(str(top_product["country_name"]))}</span>
              · {html.escape(str(top_product["shop_name"]))}</p><br>
              <span class="mp-badge mp-badge-lime">{html.escape(t(str(top_product["recommendation_label"])))}</span>
              <span class="mp-badge mp-badge-violet">{html.escape(confidence_text(str(top_product["confidence_level"])))}</span>
              <span class="mp-badge mp-badge-teal">AI: {html.escape(t(str(top_product["ai_benchmark_signal"])))}</span>
              <div class="mp-score">{top_product["opportunity_score"]:.2f}<small> / 100</small></div>
              {reasons}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="mp-card mp-dark-card">
              <span class="mp-kicker" style="color:#c9ff4a">{html.escape(t("THE OPERATING IDEA"))}</span>
              <h3>{html.escape(t("From signal overload to a review queue"))}</h3>
              <p>{html.escape(t("The transparent score is paired with a shop-grouped, cross-validated contextual benchmark. The recommendation layer translates both into a review category and auditable reasons. A merchandiser—not the system—decides what happens next."))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("Why SKUNIVO")
    cols = st.columns(3)
    cards = [
        (
            "01",
            "Context-aware benchmarking",
            "Compares products within country and platform category, keeping local market context intact.",
        ),
        (
            "02",
            "Explainable prioritization",
            "Pairs each ranked listing with peer percentiles, a recommendation type, and three traceable reasons.",
        ),
        (
            "03",
            "Human-in-the-loop decisions",
            "Creates a consistent review queue while keeping commercial judgment and testing with the team.",
        ),
    ]
    for column, (number, title, copy) in zip(cols, cards):
        with column:
            st.markdown(
                f"""
                <div class="mp-card">
                  <span class="mp-kicker">{number}</span>
                  <h3>{html.escape(t(title))}</h3><p>{html.escape(t(copy))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header("What the system does not do")
    st.markdown(
        f"""
        <div class="mp-card">
          <span class="mp-badge">{html.escape(t("No demand forecast"))}</span>
          <span class="mp-badge">{html.escape(t("No causal promotion claim"))}</span>
          <span class="mp-badge">{html.escape(t("No profit optimization"))}</span>
          <span class="mp-badge">{html.escape(t("No automatic action execution"))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_boundary()

    section_header(
        "Demo Guide",
        "A focused three-step walkthrough from ranked queue to accountable human decision.",
    )
    st.button(
        t("Start 3-minute judge demo"),
        type="primary",
        disabled=st.session_state.get("demo_guide_started", False),
        on_click=start_demo_guide,
    )
    if st.session_state.get("demo_guide_started", False):
        for index, (key, label) in enumerate(DEMO_STEPS, start=1):
            icon = "✓" if st.session_state.get(key, False) else "○"
            st.markdown(f"**{icon} {t('Step')} {index}:** {t(label)}")


def executive_page(products: pd.DataFrame, charts: dict[str, Path]) -> None:
    page_header("Executive Overview")
    market = st.segmented_control(
        t("Market view"),
        ["All markets", "Indonesia", "Vietnam"],
        default="All markets",
        selection_mode="single",
        format_func=t,
    )
    market = market or "All markets"
    code = {"Indonesia": "id", "Vietnam": "vn"}.get(market)
    view = products if code is None else products[products["country_code"].eq(code)]

    if market == "All markets":
        st.info(t("All-market view uses counts and normalized scores only. Raw IDR and VND prices are not combined or compared."))

    kpi_cols = st.columns(6)
    kpis = [
        ("Latest listings", f"{len(view):,}"),
        ("Shops", f"{view['shop_id'].nunique():,}"),
        ("Markets", f"{view['country_code'].nunique():,}"),
        ("Median score", f"{view['opportunity_score'].median():.1f}"),
        ("Protect Hero SKU", f"{view['recommendation_label'].eq('Protect Hero SKU').sum():,}"),
        ("Active review", f"{active_review_count(view):,}"),
    ]
    for column, (label, value) in zip(kpi_cols, kpis):
        column.metric(t(label), value)

    section_header("Portfolio shape", "Normalized opportunity scores and recommendation mix for the selected view.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        if market == "All markets" and st.session_state.get("language") == "vi":
            score_chart = products[["country_code", "opportunity_score"]].copy()
            score_chart["Score band"] = pd.cut(
                score_chart["opportunity_score"],
                bins=list(range(0, 105, 5)),
                include_lowest=True,
            ).astype(str)
            score_chart = (
                score_chart.groupby(["Score band", "country_code"], observed=True)
                .size()
                .unstack(fill_value=0)
                .rename(columns={"id": "Indonesia", "vn": "Vietnam"})
                .reset_index()
            )
            st.bar_chart(
                localized_frame(score_chart),
                x=t("Score band"),
                y=["Indonesia", "Vietnam"],
                height=360,
            )
            st.caption(t("Opportunity score distribution by market. Scores are normalized within local peer groups."))
        elif market == "All markets":
            show_image_chart(
                charts["score_distribution"],
                t("Opportunity score distribution by market. Scores are normalized within local peer groups."),
            )
        else:
            bins = pd.cut(
                view["opportunity_score"],
                bins=list(range(0, 105, 5)),
                include_lowest=True,
            )
            histogram = (
                bins.value_counts(sort=False)
                .rename_axis("Score band")
                .reset_index(name="Listings")
            )
            # Altair cannot serialize pandas Interval values. String labels
            # keep the market-specific chart stable across library versions.
            histogram["Score band"] = histogram["Score band"].astype(str)
            histogram_display = localized_frame(histogram)
            st.bar_chart(
                histogram_display,
                x=t("Score band"),
                y=t("Listings"),
                color="#8b7cff",
                height=360,
            )
            st.caption(t(f"Opportunity score distribution for {market}; 5-point score bands."))
    with chart_right:
        if market == "All markets" and st.session_state.get("language") == "vi":
            recommendation_chart = (
                products.groupby(["recommendation_label", "country_code"], observed=True)
                .size()
                .unstack(fill_value=0)
                .rename(columns={"id": "Indonesia", "vn": "Vietnam"})
                .reset_index()
                .rename(columns={"recommendation_label": "Recommendation"})
            )
            st.bar_chart(
                localized_frame(recommendation_chart),
                x=t("Recommendation"),
                y=["Indonesia", "Vietnam"],
                height=360,
            )
            st.caption(t("Count of listings by recommendation type and market."))
        elif market == "All markets":
            show_image_chart(
                charts["recommendation_distribution"],
                t("Count of listings by recommendation type and market."),
            )
        else:
            mix = recommendation_mix(view)
            mix_display = localized_frame(mix)
            st.bar_chart(
                mix_display,
                x=t("Recommendation"),
                y=t("Listings"),
                color="#11b8a5" if code == "id" else "#f28b30",
                height=360,
            )
            st.caption(t(f"Recommendation-label distribution for {market}."))

    table_left, table_right = st.columns(2)
    with table_left:
        st.markdown(f"#### {t('Recommendation share')}")
        mix = recommendation_mix(view)
        mix["Share of listings"] = mix["Share"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            localized_frame(mix[["Recommendation", "Listings", "Share of listings"]]),
            hide_index=True,
            width="stretch",
        )
    with table_right:
        st.markdown(f"#### {t('Score-band summary')}")
        st.dataframe(localized_frame(score_band_summary(view)), hide_index=True, width="stretch")

    section_header("Top opportunity products by market")
    if market == "All markets" and st.session_state.get("language") == "vi":
        top_all = (
            products.sort_values(["country_code", "opportunity_score"], ascending=[True, False])
            .groupby("country_code", observed=True)
            .head(5)[
                ["country_name", "product_name", "shop_name", "opportunity_score", "recommendation_label"]
            ]
            .rename(
                columns={
                    "country_name": "Country",
                    "product_name": "Product",
                    "shop_name": "Shop",
                    "opportunity_score": "Opportunity score",
                    "recommendation_label": "Recommendation",
                }
            )
        )
        st.dataframe(localized_frame(top_all), hide_index=True, width="stretch")
        st.caption(t("The highest balanced scores are shown separately for Indonesia and Vietnam."))
    elif market == "All markets":
        show_image_chart(
            charts["top_opportunities"],
            t("The highest balanced scores are shown separately for Indonesia and Vietnam."),
        )
    else:
        top = view.nlargest(10, "opportunity_score")[
            ["product_name", "shop_name", "opportunity_score", "recommendation_label"]
        ].copy()
        top.insert(0, "Rank", range(1, len(top) + 1))
        st.dataframe(localized_frame(top), hide_index=True, width="stretch")

    section_header("Executive readout")
    insights = [
        "Deep discounts do not guarantee stronger peer-relative response; discount depth should not be treated as a standalone action trigger.",
        "Engagement and rating volume are influential prioritization signals, but rating volume may also reflect listing maturity.",
        "Market context materially changes the recommendation mix, which is why products are benchmarked within country and platform category.",
        "Vietnam’s actionable benchmark shows limited ranking quality, so its recommendations require stronger human review.",
    ]
    st.markdown(
        '<div class="mp-insight">'
        + "".join(f"<div class='mp-reason'>{html.escape(t(item))}</div>" for item in insights)
        + "</div>",
        unsafe_allow_html=True,
    )
    render_boundary()


def _localized_selectbox(container, label: str, options: list[str], key: str, **kwargs):
    labels = [t(option) for option in options]
    current = st.session_state.get(key)
    if current is not None:
        for option in options:
            if current in {option, translate(option, "vi")}:
                st.session_state[key] = t(option)
                break
    selected_label = container.selectbox(t(label), labels, key=key, **kwargs)
    return options[labels.index(selected_label)]


def prioritization_page(products: pd.DataFrame) -> None:
    mark_demo_step("demo_step_prioritization")
    page_header("Product Prioritization")
    st.caption(t("Start with one market. Open advanced filters only when the decision requires them."))

    if "filter_country" not in st.session_state:
        st.session_state.filter_country = "id"
    country = st.segmented_control(
        t("Decision market"),
        ["id", "vn"],
        key="filter_country",
        selection_mode="single",
        format_func=lambda value: {"id": "Indonesia", "vn": "Vietnam"}[value],
    ) or "id"
    countries = [country]
    scoped = products[products["country_code"].eq(country)]
    st.caption(
        t("Indonesia is the recommended pilot market.")
        if country == "id"
        else t("Vietnam remains available with lower AI-model confidence and stronger human review.")
    )

    with st.expander(t("Advanced filters"), expanded=False):

        row1 = st.columns(3)
        shops = row1[0].multiselect(
            t("Shop"),
            sorted(scoped["shop_name"].dropna().astype(str).unique()),
            key="filter_shops",
        )
        platform_categories = row1[1].multiselect(
            t("Platform category"),
            sorted(scoped["platform_category"].dropna().astype(str).unique()),
            key="filter_platform_categories",
        )
        shop_categories = row1[2].multiselect(
            t("Shop category"),
            sorted(scoped["shop_category"].dropna().astype(str).unique())
            if "shop_category" in scoped
            else [],
            key="filter_shop_categories",
        )

        row2 = st.columns(3)
        recommendation_labels = row2[0].multiselect(
            t("Recommendation"),
            sorted(scoped["recommendation_label"].dropna().astype(str).unique()),
            key="filter_recommendations",
            format_func=t,
        )
        confidence_levels = row2[1].multiselect(
            t("Confidence"),
            ["High", "Medium", "Low"],
            key="filter_confidence",
            format_func=t,
        )
        promoted_status = _localized_selectbox(
            row2[2], "Promoted status", ["All", "Yes", "No"], "filter_promoted"
        )
        official_status = _localized_selectbox(
            row2[2], "Official-shop status", ["All", "Yes", "No"], "filter_official"
        )

        row3 = st.columns(4)
        score_min, score_max = safe_range(scoped["opportunity_score"], (0.0, 100.0))
        score_range = row3[0].slider(
            t("Opportunity score"),
            0.0,
            100.0,
            (
                float(max(0.0, math.floor(score_min))),
                float(min(100.0, math.ceil(score_max))),
            ),
            step=1.0,
            key="filter_score",
        )
        discount_min, discount_max = safe_range(scoped["displayed_discount_pct"])
        discount_floor = float(math.floor(discount_min))
        discount_ceiling = float(max(math.ceil(discount_max), math.floor(discount_min) + 1))
        discount_range = row3[1].slider(
            t("Discount percent"),
            discount_floor,
            discount_ceiling,
            (discount_floor, discount_ceiling),
            step=1.0,
            key="filter_discount",
        )
        likes_min, likes_max = safe_range(scoped["liked_count"])
        likes_floor = int(likes_min)
        likes_ceiling = max(int(likes_max), likes_floor + 1)
        likes_range = row3[2].slider(
            t("Likes"),
            likes_floor,
            likes_ceiling,
            (likes_floor, likes_ceiling),
            key="filter_likes",
        )
        rating_min, rating_max = safe_range(scoped["rating_count"])
        rating_floor = int(rating_min)
        rating_ceiling = max(int(rating_max), rating_floor + 1)
        rating_range = row3[3].slider(
            t("Ratings"),
            rating_floor,
            rating_ceiling,
            (rating_floor, rating_ceiling),
            key="filter_rating_count",
        )

        price_range = None
        if not scoped.empty:
            price_min, price_max = safe_range(scoped["price"])
            currency = format_local_price(0, country).split()[0]
            price_range = st.slider(
                f"{t('Current price')} · {t('local')} {currency} {t('units')}",
                int(price_min),
                max(int(price_max), int(price_min) + 1),
                (int(price_min), max(int(price_max), int(price_min) + 1)),
                key=f"filter_price_{country}",
            )

    filters = {
        "countries": countries,
        "shops": shops,
        "platform_categories": platform_categories,
        "shop_categories": shop_categories,
        "recommendation_labels": recommendation_labels,
        "confidence_levels": confidence_levels,
        "score_range": score_range,
        "discount_range": discount_range,
        "likes_range": likes_range,
        "rating_count_range": rating_range,
        "price_range": price_range,
        "promoted_status": promoted_status,
        "official_status": official_status,
    }
    filtered = apply_product_filters(products, filters)

    controls = st.columns([2, 1, 1])
    sort_option = _localized_selectbox(
        controls[0], "Sort results", list(SORT_OPTIONS), "sort_products"
    )
    row_limit = controls[1].selectbox(t("Rows per page"), [25, 50, 100], index=0)
    total_pages = max(1, math.ceil(len(filtered) / row_limit))
    page_number = controls[2].number_input(
        t("Page"), min_value=1, max_value=total_pages, value=1, step=1
    )
    filtered = sort_products(filtered, sort_option)

    top_label = (
        filtered["recommendation_label"].mode().iloc[0] if not filtered.empty else "Not available"
    )
    metric_cards(
        [
            ("Filtered products", f"{len(filtered):,}"),
            (
                "Median score",
                f"{filtered['opportunity_score'].median():.1f}" if not filtered.empty else "—",
            ),
            ("Top recommendation", t(str(top_label))),
            ("Active-review candidates", f"{active_review_count(filtered):,}"),
        ]
    )

    if filtered.empty:
        st.info(t("No products match the current filters. Broaden one or more ranges or categories."))
        render_boundary()
        return

    start = (int(page_number) - 1) * row_limit
    page_view = filtered.iloc[start : start + row_limit].copy()
    page_view.insert(0, "Rank", range(start + 1, start + 1 + len(page_view)))
    page_view["Country"] = page_view["country_code"].map({"id": "Indonesia", "vn": "Vietnam"})
    page_view["Product"] = page_view["product_name"].map(
        lambda value: str(value)[:78] + ("…" if len(str(value)) > 78 else "")
    )
    page_view["Current price"] = page_view.apply(
        lambda row: format_local_price(row["price"], row["country_code"]), axis=1
    )
    table = page_view[
        [
            "Rank",
            "Country",
            "shop_name",
            "Product",
            "platform_category",
            "Current price",
            "displayed_discount_pct",
            "liked_count",
            "rating_count",
            "monthly_sold_value",
            "opportunity_score",
            "recommendation_label",
            "confidence_level",
        ]
    ].rename(
        columns={
            "shop_name": "Shop",
            "platform_category": "Category",
            "displayed_discount_pct": "Discount %",
            "liked_count": "Likes",
            "rating_count": "Rating count",
            "monthly_sold_value": "Monthly sold-value proxy",
            "opportunity_score": "Opportunity score",
            "recommendation_label": "Recommendation",
            "confidence_level": "Confidence",
        }
    )
    table_display = localized_frame(table)
    st.dataframe(
        table_display,
        hide_index=True,
        width="stretch",
        height=min(680, 72 + 35 * len(table)),
        column_config={
            t("Opportunity score"): st.column_config.ProgressColumn(
                t("Opportunity score"), min_value=0, max_value=100, format="%.2f"
            ),
            t("Discount %"): st.column_config.NumberColumn(t("Discount %"), format="%.1f%%"),
        },
    )
    if st.session_state.get("language") == "vi":
        st.caption(
            f"Đang hiển thị {start + 1:,}–{start + len(page_view):,} trên "
            f"{len(filtered):,} sản phẩm. {t('Prices are labeled in each listing’s local market units.')}"
        )
    else:
        st.caption(
            f"Showing {start + 1:,}–{start + len(page_view):,} of {len(filtered):,} products. "
            "Prices are labeled in each listing’s local market units."
        )

    download = filtered.copy()
    download["market_currency"] = download["country_code"].map({"id": "IDR", "vn": "VND"})
    st.download_button(
        t("Download current filtered results"),
        download.to_csv(index=False).encode("utf-8-sig"),
        "skunivo_filtered_products.csv",
        "text/csv",
    )

    section_header("Inspect a product")
    lookup = dict(
        zip(
            filtered["product_key"],
            filtered.apply(
                lambda row: f"{row['country_code'].upper()} · {row['item_id']} · {str(row['product_name'])[:90]}",
                axis=1,
            ),
        )
    )
    selected_key = st.selectbox(
        t("Select from the filtered queue"),
        list(lookup),
        format_func=lambda key: lookup[key],
        label_visibility="collapsed",
    )
    st.button(
        t("Open product explanation →"),
        type="primary",
        on_click=open_product_explanation,
        args=(selected_key,),
    )
    render_boundary()


def product_explanation_page(products: pd.DataFrame) -> None:
    mark_demo_step("demo_step_explanation")
    page_header("Product Explanation")
    countries = ["id", "vn"]
    remembered = st.session_state.get("selected_product_key")
    remembered_row = (
        products[products["product_key"].eq(remembered)].iloc[0]
        if remembered and products["product_key"].eq(remembered).any()
        else None
    )
    default_country = remembered_row["country_code"] if remembered_row is not None else "id"

    selectors = st.columns(2)
    country = selectors[0].selectbox(
        t("Country"),
        countries,
        index=countries.index(default_country),
        format_func=lambda value: {"id": "Indonesia", "vn": "Vietnam"}[value],
        key="explain_country",
    )
    country_view = products[products["country_code"].eq(country)]
    shop_options = sorted(country_view["shop_name"].dropna().unique())
    default_shop = (
        remembered_row["shop_name"]
        if remembered_row is not None and remembered_row["country_code"] == country
        else shop_options[0]
    )
    shop = selectors[1].selectbox(
        t("Shop"), shop_options, index=shop_options.index(default_shop), key="explain_shop"
    )
    shop_view = country_view[country_view["shop_name"].eq(shop)]
    search = st.text_input(
        t("Search by item ID or product name"),
        placeholder=t("Type part of a product name or an item ID"),
        key="explain_search",
    ).strip()
    matches = shop_view
    if search:
        match_mask = (
            matches["product_name"].astype(str).str.contains(search, case=False, regex=False, na=False)
            | matches["item_id"].astype(str).str.contains(search, regex=False, na=False)
        )
        matches = matches[match_mask]
    if matches.empty:
        st.info(t("No product in this shop matches the search. Try a broader phrase or clear the search."))
        return

    option_keys = matches["product_key"].tolist()
    default_key = remembered if remembered in option_keys else option_keys[0]
    selected_key = st.selectbox(
        t("Item ID and product"),
        option_keys,
        index=option_keys.index(default_key),
        format_func=lambda key: (
            f"{matches.loc[matches['product_key'].eq(key), 'item_id'].iloc[0]} · "
            f"{str(matches.loc[matches['product_key'].eq(key), 'product_name'].iloc[0])[:110]}"
        ),
    )
    product = products[products["product_key"].eq(selected_key)].iloc[0]
    st.session_state.selected_product_key = selected_key

    market_badge = "mp-badge-teal" if country == "id" else "mp-badge-orange"
    st.markdown(
        f"""
        <div class="mp-preview">
          <span class="mp-kicker">{html.escape(t("PRODUCT DECISION RECORD"))}</span>
          <h2>{html.escape(str(product["product_name"]))}</h2>
          <p>{html.escape(t("Item"))} {html.escape(str(product["item_id"]))} · {html.escape(str(product["shop_name"]))}</p><br>
          <span class="mp-badge {market_badge}">{html.escape(str(product["country_name"]))}</span>
          <span class="mp-badge">{html.escape(str(product["platform_category"]))}</span>
          <span class="mp-badge mp-badge-lime">{html.escape(t(str(product["recommendation_label"])))}</span>
          <span class="mp-badge mp-badge-violet">{html.escape(confidence_text(str(product["confidence_level"])))}</span>
          <div class="mp-score">{product["opportunity_score"]:.2f}<small> {html.escape(t("opportunity score / 100"))}</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header(
        "Decision at a glance",
        "Read the recommendation, reasons, and next action first. Technical evidence remains available below.",
    )
    reason_column, action_column = st.columns([1.05, 0.95])
    with reason_column:
        st.markdown(f"### {t('Why this recommendation')}")
        for reason_name in ("reason_1", "reason_2", "reason_3"):
            reason = product.get(reason_name)
            if reason is not None and not pd.isna(reason):
                st.markdown(
                    f'<div class="mp-reason">{html.escape(t(str(reason)))}</div>',
                    unsafe_allow_html=True,
                )

    guidance = GUIDANCE.get(
        str(product["recommendation_label"]),
        ["Review the observed signals with a merchandiser."],
    )
    with action_column:
        st.markdown(f"### {t('Suggested next action')}")
        st.markdown(
            '<div class="mp-card">'
            + "".join(f"<div class='mp-reason'>{html.escape(t(item))}</div>" for item in guidance)
            + f"<p style='margin-top:1rem'>{html.escape(t('These are review prompts, not mandatory actions or outcome guarantees.'))}</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            t("Open Decision Log →"),
            type="primary",
            width="stretch",
            on_click=open_decision_log,
            args=(selected_key,),
        )

    with st.expander(t("Supporting evidence and model details"), expanded=False):
        section_header("Observed product and shop signals")
        metrics = [
            ("Current price", format_local_price(product["price"], country)),
            ("Original price", format_local_price(product.get("original_price"), country)),
            ("Discount", f"{product['displayed_discount_pct']:.1f}%"),
            ("Likes", format_number(product["liked_count"])),
            ("Product rating", display_value(product.get("rating_star"), digits=2)),
            ("Rating count", format_number(product["rating_count"])),
            ("Monthly sold-value proxy", format_number(product["monthly_sold_value"], 1)),
            ("Shop rating", display_value(product.get("shop_rating"), digits=2)),
            ("Shop followers", format_number(product.get("shop_follower_count"))),
            ("Official shop", bool_label(product.get("is_official_shop"))),
        ]
        cols = st.columns(5)
        for index, (label, value) in enumerate(metrics):
            cols[index % 5].metric(t(label), t(value) if isinstance(value, str) else value)

        section_header(
            "Peer-relative benchmarks",
            "Percentiles are calculated within country and platform category; higher is not automatically better for every component.",
        )
        peer_metrics = [
            ("Engagement peer percentile", format_percent(product.get("likes_pct_peer"))),
            ("Sold-value peer percentile", format_percent(product.get("sold_pct_peer"))),
            ("Price peer percentile", format_percent(product.get("price_pct_country_category"))),
            ("Discount peer percentile", format_percent(product.get("discount_pct_peer"))),
            ("Shop credibility", format_percent(product.get("shop_credibility"))),
            ("Conversion-gap component", format_percent(product.get("conversion_gap"))),
        ]
        cols = st.columns(3)
        for index, (label, value) in enumerate(peer_metrics):
            cols[index % 3].metric(t(label), value)

        section_header(
            "AI-assisted contextual benchmark",
            "A shop-grouped, cross-validated model estimates the sold-value level associated with the current listing context. It is not a future-sales forecast.",
        )
        expected = product.get("ai_contextual_sold_benchmark")
        observed = product.get("monthly_sold_value")
        gap = product.get("ai_benchmark_gap_pct")
        model_confidence = str(product.get("ai_model_confidence", "Unavailable"))
        benchmark_metrics = st.columns(4)
        benchmark_metrics[0].metric(
            t("Model benchmark"),
            format_number(expected, 1),
            help=t("Cross-validated contextual sold-value proxy estimate."),
        )
        benchmark_metrics[1].metric(t("Observed proxy"), format_number(observed, 1))
        benchmark_metrics[2].metric(
            t("Observed gap"),
            t("Not available") if gap is None or pd.isna(gap) else f"{float(gap):+.0%}",
        )
        benchmark_metrics[3].metric(t("Model confidence"), t(model_confidence))
        signal_class = "mp-badge-teal" if model_confidence == "High" else "mp-badge-orange"
        st.markdown(
            f"""
            <div class="mp-card">
              <span class="mp-kicker">{html.escape(t("AI DECISION BRIEF"))}</span><br>
              <span class="mp-badge {signal_class}">{html.escape(t(str(product.get("ai_benchmark_signal", "Unavailable"))))}</span>
              <span class="mp-badge">{html.escape(confidence_text(model_confidence, model=True))}</span>
              <p style="margin-top:1rem">{html.escape(ai_decision_brief(product, st.session_state.get("language", "en")))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        local_drivers = product.get("ai_local_drivers")
        if local_drivers is not None and not pd.isna(local_drivers):
            with st.expander(t("Model contribution detail for this representative product")):
                for driver in str(local_drivers).split(" | "):
                    st.markdown(f"- {t(driver)}")
        if country == "vn":
            st.warning(t("Vietnam actionable-model ranking quality is limited. Treat this benchmark as supporting evidence only; transparent scoring and human review take priority."))

    export_fields = {
        "country": product["country_name"],
        "shop_name": product["shop_name"],
        "item_id": product["item_id"],
        "product_name": product["product_name"],
        "category": product["platform_category"],
        "opportunity_score": product["opportunity_score"],
        "recommendation_label": product["recommendation_label"],
        "confidence_level": product["confidence_level"],
        "reason_1": product["reason_1"],
        "reason_2": product["reason_2"],
        "reason_3": product["reason_3"],
        "decision_boundary": BOUNDARY,
    }
    st.download_button(
        t("Download Product Decision Summary"),
        pd.DataFrame([export_fields]).to_csv(index=False).encode("utf-8-sig"),
        f"skunivo_product_{product['item_id']}.csv",
        "text/csv",
    )
    render_boundary()


def decision_log_page(products: pd.DataFrame) -> None:
    page_header("Decision Log")
    remembered = st.session_state.get("selected_product_key")
    if remembered and products["product_key"].eq(remembered).any():
        default_key = remembered
    else:
        default_key = products.sort_values("opportunity_score", ascending=False).iloc[0]["product_key"]
    product_keys = products["product_key"].tolist()
    selected_key = st.selectbox(
        t("Product decision record"),
        product_keys,
        index=product_keys.index(default_key),
        format_func=lambda key: (
            f"{products.loc[products['product_key'].eq(key), 'country_code'].iloc[0].upper()} · "
            f"{products.loc[products['product_key'].eq(key), 'item_id'].iloc[0]} · "
            f"{str(products.loc[products['product_key'].eq(key), 'product_name'].iloc[0])[:95]}"
        ),
        key="decision_product_key",
    )
    product = products[products["product_key"].eq(selected_key)].iloc[0]
    st.session_state.selected_product_key = selected_key

    st.markdown(
        f"""
        <div class="mp-preview">
          <span class="mp-kicker">{html.escape(t("HUMAN DECISION RECORD"))}</span>
          <h2>{html.escape(str(product["product_name"]))}</h2>
          <p>{html.escape(str(product["shop_name"]))} · {html.escape(t("Item"))} {html.escape(str(product["item_id"]))}</p><br>
          <span class="mp-badge mp-badge-lime">{html.escape(t(str(product["recommendation_label"])))}</span>
          <span class="mp-badge">{product["opportunity_score"]:.2f} {html.escape(t("Opportunity score").lower())}</span>
          <span class="mp-badge">{html.escape(t(str(product.get("ai_benchmark_signal", "Benchmark unavailable"))))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    webhook_url, webhook_token = google_sheets_config()
    if webhook_url:
        st.success(t("Decision persistence is connected to Team YOUNGHTT's Google Sheet."))
    else:
        st.info(t("Google Sheets is not configured yet. Local runs append to outputs/mvp_decision_log.csv; public submissions remain downloadable until the webhook is added."))

    with st.form("decision_log_form", clear_on_submit=False):
        cols = st.columns(2)
        reviewer_role = cols[0].selectbox(
            t("Reviewer role *"),
            ["", "Merchandiser", "Category manager", "Commercial lead", "Judge or mentor", "Other"],
            format_func=t,
        )
        reviewer_name = cols[1].text_input(t("Reviewer name (optional)"))
        decision_status = st.radio(
            t("Decision *"),
            ["", "Accept recommendation", "Override recommendation", "Need more evidence"],
            horizontal=True,
            format_func=t,
        )
        selected_action = st.selectbox(
            t("Action *"),
            [
                "",
                "Protect current execution",
                "Review price",
                "Review discount efficiency",
                "Inspect conversion friction",
                "Prepare controlled promotion test",
                "Maintain and monitor",
                "Deprioritize immediate action",
                "Other",
            ],
            format_func=t,
        )
        decision_rationale = st.text_area(
            t("Decision rationale *"),
            placeholder=t("What evidence supports this decision, and why are you accepting or overriding the recommendation?"),
        )
        success_metric = st.text_input(
            t("Success metric *"),
            value=t("Peer-relative sold-value and engagement change"),
        )
        review_date = st.date_input(
            t("Review date *"),
            value=date.today() + timedelta(days=14),
            min_value=date.today(),
        )
        submitted = st.form_submit_button(t("Save decision"), type="primary")

    if submitted:
        row = make_decision_row(
            product,
            {
                "reviewer_role": reviewer_role,
                "reviewer_name": reviewer_name,
                "decision_status": decision_status,
                "selected_action": selected_action,
                "decision_rationale": decision_rationale,
                "success_metric": success_metric,
                "review_date": review_date.isoformat(),
            },
        )
        missing = validate_decision(row)
        if missing:
            st.error(t("Please complete every field marked with an asterisk before saving."))
        else:
            st.session_state.setdefault("decision_submissions", []).append(row)
            public_mode = is_public_mode(getattr(st.context, "url", None))
            local_saved = False
            if not public_mode:
                try:
                    append_decision(row, ROOT / "outputs" / "mvp_decision_log.csv")
                    local_saved = True
                except OSError:
                    logging.exception("Local decision append failed")
            delivery = deliver_record(
                "decision",
                row,
                webhook_url=webhook_url,
                webhook_token=webhook_token,
            )
            if delivery.delivered:
                st.success(t("Decision saved to Team YOUNGHTT's decision log."))
            elif local_saved:
                st.success(t("Decision appended to the local decision log."))
            else:
                st.warning(t("Persistent storage is not configured or unavailable. Download this decision record so it is not lost."))
                st.download_button(
                    t("Download decision record"),
                    decision_csv_bytes([row]),
                    f"skunivo_decision_{product['item_id']}.csv",
                    "text/csv",
                )
            mark_demo_step("demo_step_decision")
    render_boundary()


def _apply_preset(name: str) -> None:
    for key, value in SCORE_PRESETS[name].items():
        st.session_state[f"whatif_{key}"] = value


def what_if_page() -> None:
    page_header("What-if Score Explorer")
    st.markdown(
        f'<div class="mp-warning"><strong>{html.escape(t("Important:"))}</strong> '
        f'{html.escape(t("This simulator demonstrates how the transparent score changes. It does not predict causal promotion lift, profit, or future sales."))}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"#### {t('Component presets')}")
    preset_cols = st.columns(3)
    for column, preset in zip(preset_cols, SCORE_PRESETS):
        column.button(
            t(preset),
            on_click=_apply_preset,
            args=(preset,),
            width="stretch",
        )
    st.caption(t("Presets adjust component sliders only; they do not represent exact real-world outcomes."))

    for key in SCORE_WEIGHTS:
        if f"whatif_{key}" not in st.session_state:
            st.session_state[f"whatif_{key}"] = SCORE_PRESETS["Balanced product"][key]

    slider_left, slider_right = st.columns(2)
    components: dict[str, float] = {}
    for index, key in enumerate(SCORE_WEIGHTS):
        target = slider_left if index % 2 == 0 else slider_right
        components[key] = target.slider(
            f"{t(COMPONENT_LABELS[key])} · {SCORE_WEIGHTS[key]:.0%} {t('Weight').lower()}",
            0,
            100,
            key=f"whatif_{key}",
            help=t("A normalized component from 0 to 100."),
        )

    score = calculate_what_if_score(components)
    contributions = score_contributions(components)
    result_left, result_right = st.columns([1, 1.6])
    with result_left:
        st.markdown(
            f"""
            <div class="mp-preview">
              <span class="mp-kicker">{html.escape(t("LIVE TRANSPARENT SCORE"))}</span>
              <div class="mp-score">{score:.1f}<small> / 100</small></div>
              <span class="mp-badge mp-badge-violet">{html.escape(t("Illustrative score interpretation"))}</span>
              <h3>{html.escape(t(illustrative_tier(score)))}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_right:
        st.bar_chart(localized_frame(contribution_frame(contributions)), color="#8b7cff", horizontal=True, height=330)
        st.caption(t("Weighted contribution to the final 0–100 score."))

    ordered = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    driver_cols = st.columns(2)
    with driver_cols[0]:
        st.markdown(f"#### {t('Dominant positive drivers')}")
        for name, value in ordered[:2]:
            st.markdown(f"- {t(name)}: **{value:.1f} {t('points')}**")
    with driver_cols[1]:
        st.markdown(f"#### {t('Weak components')}")
        for name, value in ordered[-2:]:
            st.markdown(f"- {t(name)}: **{value:.1f} {t('points')}**")

    with st.expander(t("See the exact formula")):
        if st.session_state.get("language") == "vi":
            st.code(
                "điểm = 0.25×tương_tác + 0.20×giá_trị_bán + 0.10×cạnh_tranh_giá\n"
                "      + 0.15×hiệu_quả_khuyến_mãi + 0.10×uy_tín_cửa_hàng\n"
                "      + 0.20×cơ_hội_khoảng_cách_chuyển_đổi",
                language=None,
            )
        else:
            st.code(
                "score = 0.25×engagement + 0.20×sold-value + 0.10×price competitiveness\n"
                "      + 0.15×promotion efficiency + 0.10×shop credibility\n"
                "      + 0.20×conversion-gap opportunity",
                language=None,
            )
    render_boundary()


def methodology_page(data: dict) -> None:
    page_header("Methodology and Transparency")

    section_header("1 · Data scope")
    metric_cards(
        [
            ("Latest records", "1,157"),
            ("Shops", "20"),
            ("Markets", "Indonesia + Vietnam"),
            ("Snapshot dates", "2026-07-01 → 2026-07-03"),
        ]
    )
    st.markdown(
        t(
            "Five logical table families support the pipeline: products, shop information, platform categories, "
            "product-category mappings, and shop category lists. **30 exact duplicate rows were removed.**"
        )
    )

    section_header("2 · Data processing")
    if st.session_state.get("language") == "vi":
        st.markdown(
            """
            - Chọn bản ghi mới nhất của từng sản phẩm theo quốc gia, cửa hàng và mã sản phẩm.
            - Loại bỏ các ảnh chụp dữ liệu trùng lặp hoàn toàn trước khi xếp hạng.
            - Kết nối dữ liệu sản phẩm, cửa hàng và ngành hàng bằng khóa có xét quốc gia.
            - Chuẩn hóa tín hiệu nhóm tương đồng trong **quốc gia + ngành hàng nền tảng**.
            - Giữ IDR và VND theo giá trị thị trường địa phương; không so sánh trực tiếp giá thô.
            - Giữ nguyên giới hạn ánh xạ ngành hàng cửa hàng: chỉ **66,1%** sản phẩm mới nhất có ngành hàng cửa hàng.
            """
        )
    else:
        st.markdown(
            """
            - Select the latest observed listing per country, shop, and item.
            - Remove exact duplicate snapshots before ranking.
            - Join product, shop, and category data with country-aware keys.
            - Normalize peer signals within **country + platform category**.
            - Keep IDR and VND as local-market values; never directly compare raw prices.
            - Preserve the shop-category limitation: only **66.1%** of latest listings map to a shop category.
            """
        )

    section_header("3 · Transparent scoring")
    weights = pd.DataFrame(
        {
            "Component": [COMPONENT_LABELS[key] for key in SCORE_WEIGHTS],
            "Weight": list(SCORE_WEIGHTS.values()),
        }
    )
    weights["Weight"] = weights["Weight"].map(lambda value: f"{value:.0%}")
    st.dataframe(localized_frame(weights), hide_index=True, width="stretch")
    st.markdown(
        t(
            "The balanced score is the production decision core. All six components are normalized to 0–100 "
            "and combined with the displayed weights."
        )
    )

    section_header("4 · ML-assisted benchmark")
    if st.session_state.get("language") == "vi":
        st.markdown(
            """
            Thử nghiệm **bối cảnh có thể hành động** loại bỏ giá trị bán lịch sử. Thử nghiệm
            **mô tả** riêng biệt có sử dụng giá trị bán lịch sử và được xác định rõ là có nguy cơ
            rò rỉ dữ liệu nếu dùng cho dự báo tương lai. Quá trình đánh giá nhóm sản phẩm theo cửa hàng,
            sử dụng kiểm định chéo năm phần theo nhóm và kiểm định bỏ lần lượt một cửa hàng.
            ML là đối chuẩn giải thích, không phải lớp ra quyết định chính thức.
            """
        )
    else:
        st.markdown(
            """
            The **actionable-context** experiment excludes historical sold value. The separate
            **descriptive** experiment includes historical sold value and is explicitly leakage-prone
            for future prediction. Evaluation groups products by shop using grouped five-fold and
            leave-one-shop-out validation. ML is an explanatory benchmark—not the production decision layer.
            """
        )

    section_header("5 · Model results")
    result_cols = st.columns(2)
    with result_cols[0]:
        st.markdown(
            f"""
            <div class="mp-card">
              <span class="mp-badge mp-badge-teal">Indonesia · {html.escape("có thể hành động" if st.session_state.get("language") == "vi" else "actionable")}</span>
              <h3>{html.escape(t("Materially stronger than the dummy baseline"))}</h3>
              <p>{"RMSE theo nhóm 1,196 so với đường cơ sở 2,212 (cải thiện 45,9%);" if st.session_state.get("language") == "vi" else "Grouped RMSE 1.196 vs 2.212 baseline (45.9% improvement);"}
              Spearman 0.767; top-decile lift 1.701; NDCG@20 1.000.
              {"Spearman khi bỏ lần lượt một cửa hàng là 0,770." if st.session_state.get("language") == "vi" else "Leave-one-shop-out Spearman is 0.770."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_cols[1]:
        st.markdown(
            f"""
            <div class="mp-card">
              <span class="mp-badge mp-badge-orange">Vietnam · {html.escape("có thể hành động" if st.session_state.get("language") == "vi" else "actionable")}</span>
              <h3>{html.escape(t("Limited ranking quality"))}</h3>
              <p>{"RMSE theo nhóm 2,057 so với đường cơ sở 2,309 (cải thiện 10,9%);" if st.session_state.get("language") == "vi" else "Grouped RMSE 2.057 vs 2.309 baseline (10.9% improvement);"}
              Spearman 0.407; top-decile lift 0.734; NDCG@20 0.045.
              {"Kết quả chưa đạt quy tắc cải thiện đáng kể nghiêm ngặt hơn." if st.session_state.get("language") == "vi" else "This does not meet the stricter material-improvement rule."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.info(t("Transparent scoring remains the decision core. Vietnam recommendations require stronger human review."))
    if st.session_state.get("language") == "vi":
        model_chart = data["model_metrics"]
        model_chart = model_chart[
            model_chart["experiment"].eq("actionable_context")
            & model_chart["validation_scheme"].eq("group_5_fold")
        ][["model", "country_code", "rmse_log"]]
        model_chart = (
            model_chart.pivot(index="model", columns="country_code", values="rmse_log")
            .rename(columns={"id": "Indonesia", "vn": "Vietnam"})
            .reset_index()
            .rename(columns={"model": "Mô hình"})
        )
        st.bar_chart(model_chart, x="Mô hình", y=["Indonesia", "Vietnam"], height=360)
        st.caption(t("Actionable benchmark performance relative to the dummy baseline. Metrics are evaluation diagnostics, not outcome guarantees."))
    else:
        show_image_chart(
            data["charts"]["model_baseline"],
            t("Actionable benchmark performance relative to the dummy baseline. Metrics are evaluation diagnostics, not outcome guarantees."),
        )

    section_header("6 · Feature importance")
    if st.session_state.get("language") == "vi":
        importance_chart = (
            data["feature_importance"]
            .query("experiment == 'actionable_context'")
            .groupby("feature_label", observed=True)["importance"]
            .mean()
            .nlargest(10)
            .sort_values()
            .reset_index()
            .rename(columns={"feature_label": "Đặc trưng", "importance": "Tầm quan trọng"})
        )
        importance_chart["Đặc trưng"] = importance_chart["Đặc trưng"].map(t)
        st.bar_chart(
            importance_chart,
            x="Tầm quan trọng",
            y="Đặc trưng",
            horizontal=True,
            height=380,
        )
        st.caption(t("Global importance from the best actionable models. Importance describes model reliance, not causality."))
    else:
        show_image_chart(
            data["charts"]["feature_importance"],
            t("Global importance from the best actionable models. Importance describes model reliance, not causality."),
        )
    st.warning(t("Rating volume dominates the model and may partly reflect listing maturity or age, which is not directly observed."))

    section_header("7 · Ranking robustness")
    if st.session_state.get("language") == "vi":
        sensitivity_chart = (
            data["score_sensitivity"]
            .pivot(
                index="configuration",
                columns="country_code",
                values="top20_jaccard_vs_balanced",
            )
            .rename(columns={"id": "Indonesia", "vn": "Vietnam"})
            .reset_index()
        )
        sensitivity_chart["configuration"] = sensitivity_chart["configuration"].map(
            {
                "balanced": t("Balanced product"),
                "growth_opportunity": t("Growth opportunity"),
                "hero_protection": t("Hero protection"),
            }
        )
        sensitivity_chart = sensitivity_chart.rename(columns={"configuration": "Cấu hình"})
        st.bar_chart(sensitivity_chart, x="Cấu hình", y=["Indonesia", "Vietnam"], height=360)
        st.caption(t("Top-20 overlap and full-ranking correlation under balanced, growth-opportunity, and hero-protection weights."))
    else:
        show_image_chart(
            data["charts"]["score_sensitivity"],
            t("Top-20 overlap and full-ranking correlation under balanced, growth-opportunity, and hero-protection weights."),
        )
    st.markdown(t(
        "Indonesia rankings are more stable. Vietnam’s top rankings are more sensitive to scoring priorities: "
        "top-20 overlap versus balanced falls to 0.538 for growth-opportunity and 0.482 for hero-protection."
    ))

    section_header("8 · Limitations")
    limitations = [
        "Only three snapshot dates",
        "Cumulative engagement signals",
        "No orders",
        "No customers",
        "No cost data",
        "No inventory history",
        "No ad spend",
        "No realized revenue",
        "No experimental treatment/control",
        "No currency metadata",
    ]
    st.markdown(
        '<div class="mp-card">'
        + "".join(f'<span class="mp-badge">{html.escape(t(item))}</span>' for item in limitations)
        + "</div>",
        unsafe_allow_html=True,
    )

    section_header("9 · Future data roadmap")
    roadmap_cols = st.columns(4)
    roadmap = [
        ("Current", "Explainable product prioritization"),
        ("Next", "Controlled promotion testing"),
        ("Later", "Causal lift measurement"),
        ("Future", "Demand forecasting, margin-aware optimization, and inventory planning"),
    ]
    for column, (stage, content) in zip(roadmap_cols, roadmap):
        column.markdown(
            f'<div class="mp-card"><span class="mp-kicker">{html.escape(t(stage))}</span><h3>{html.escape(t(content))}</h3></div>',
            unsafe_allow_html=True,
        )
    st.caption(t("Roadmap items after Current are proposed future capabilities and are not present in this MVP."))
    render_boundary()


def feedback_page() -> None:
    page_header("User Feedback")
    try:
        current_url = st.context.url
    except Exception:
        current_url = None
    public_mode = is_public_mode(current_url)
    webhook_url, webhook_token = google_sheets_config()
    if webhook_url:
        st.success(t("Feedback persistence is connected to Team YOUNGHTT's Google Sheet."))
    elif public_mode:
        st.info(t(
            "Public-session feedback is temporary because persistent external storage is not configured. "
            "A CSV fallback will be offered only until the Team YOUNGHTT webhook is added."
        ))
    else:
        st.success(t("Local mode: valid submissions append to the local feedback CSV without overwriting prior rows."))
    st.caption(t("Feedback is collected only for prototype evaluation. Personal details are optional."))

    with st.form("mvp_feedback_form", clear_on_submit=False):
        main_cols = st.columns(2)
        role = main_cols[0].selectbox(
            t("Participant role *"),
            ["", "Merchandiser", "Category manager", "Analyst", "Product manager", "Judge or mentor", "Other"],
            format_func=t,
        )
        scenario = main_cols[1].selectbox(
            t("Test scenario completed *"),
            [
                "",
                "Executive portfolio review",
                "Product shortlist creation",
                "Recommendation explanation audit",
                "What-if score exploration",
                "End-to-end demo guide",
            ],
            format_func=t,
        )
        rating_cols = st.columns(4)
        usefulness = rating_cols[0].select_slider(t("Usefulness *"), options=[1, 2, 3, 4, 5], value=3)
        clarity = rating_cols[1].select_slider(t("Explanation clarity *"), options=[1, 2, 3, 4, 5], value=3)
        trust = rating_cols[2].select_slider(t("Trust *"), options=[1, 2, 3, 4, 5], value=3)
        navigation = rating_cols[3].select_slider(t("Navigation *"), options=[1, 2, 3, 4, 5], value=3)
        would_use = st.radio(
            t("Would you use this for product review? *"),
            ["", "Yes", "No"],
            horizontal=True,
            format_func=t,
        )
        useful_feature = st.text_area(t("Most useful feature *"), height=90)
        confusing = st.text_area(t("Most confusing element *"), height=90)
        improvement = st.text_area(t("Suggested improvement *"), height=90)
        st.markdown(f"##### {t('Optional participant details')}")
        personal_cols = st.columns(3)
        participant_name = personal_cols[0].text_input(t("Name"))
        organization = personal_cols[1].text_input(t("Organization"))
        email = personal_cols[2].text_input(t("Email"))
        submitted = st.form_submit_button(t("Submit prototype feedback"), type="primary")

    if submitted:
        row = make_feedback_row(
            {
                "participant_role": role,
                "test_scenario": scenario,
                "usefulness_rating": usefulness,
                "explanation_clarity_rating": clarity,
                "trust_rating": trust,
                "navigation_rating": navigation,
                "would_use": would_use,
                "most_useful_feature": useful_feature,
                "confusing_element": confusing,
                "improvement_suggestion": improvement,
                "participant_name": participant_name,
                "organization": organization,
                "email": email,
            }
        )
        missing = validate_feedback(row)
        if missing:
            st.error(t("Please complete every field marked with an asterisk before submitting."))
        else:
            st.session_state.setdefault("feedback_submissions", []).append(row)
            local_saved = False
            if not public_mode:
                try:
                    append_feedback(row, ROOT / "outputs" / "mvp_user_feedback.csv")
                    local_saved = True
                except OSError:
                    logging.exception("Local feedback append failed")
            delivery = deliver_record(
                "feedback",
                row,
                webhook_url=webhook_url,
                webhook_token=webhook_token,
            )
            if delivery.delivered:
                st.success(t("Thank you. Your feedback was submitted to Team YOUNGHTT."))
            elif local_saved:
                st.success(t("Thank you. Your feedback was appended to the local evaluation file."))
            else:
                st.warning(t("Persistent storage is not configured or unavailable. Download this feedback row so it is not lost."))
                st.download_button(
                    t("Download submitted feedback row"),
                    feedback_csv_bytes([row]),
                    "skunivo_feedback_submission.csv",
                    "text/csv",
                )
            mark_demo_step("demo_step_feedback")
    render_boundary()


def main() -> None:
    render_shell()
    try:
        data = load_app_data()
    except DataValidationError as exc:
        logging.exception("Startup validation failed")
        st.error(t(
            "SKUNIVO could not load its precomputed decision outputs. "
            "Please confirm the deployment includes the required outputs and chart files."
        ))
        with st.expander(t("Validation detail")):
            st.write(str(exc))
        st.stop()
    except Exception:
        logging.exception("Unexpected startup failure")
        st.error(t("SKUNIVO encountered an unexpected startup problem. Technical details were logged."))
        st.stop()

    page = st.session_state.active_page
    products = data["products"]
    if page == "Home":
        home_page(products)
    elif page == "Executive Overview":
        executive_page(products, data["charts"])
    elif page == "Product Prioritization":
        prioritization_page(products)
    elif page == "Product Explanation":
        product_explanation_page(products)
    elif page == "Decision Log":
        decision_log_page(products)
    elif page == "What-if Score Explorer":
        what_if_page()
    elif page == "Methodology and Transparency":
        methodology_page(data)
    elif page == "User Feedback":
        feedback_page()
    render_footer()


if __name__ == "__main__":
    main()
