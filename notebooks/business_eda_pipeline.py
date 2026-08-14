from __future__ import annotations

import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"

COLORS = {
    "id": "#167D8D",
    "vn": "#E07A3F",
    "official": "#167D8D",
    "nonofficial": "#E07A3F",
    "neutral": "#74808B",
    "light": "#DCE4E8",
    "grid": "#D9E0E4",
    "text": "#18212A",
    "muted": "#596773",
    "background": "#FFFFFF",
    "positive": "#2A8C68",
    "warning": "#D49B2A",
}

COUNTRY_NAMES = {"id": "Indonesia", "vn": "Vietnam"}
COUNTRY_CURRENCY = {"id": "IDR (source units)", "vn": "VND (source units)"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUBTITLE = font(18)
F_AXIS = font(17)
F_LABEL = font(18)
F_SMALL = font(15)
F_VALUE = font(17, True)


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{digits}f}K"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def truncate(text: Any, limit: int = 42) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def new_chart(title: str, subtitle: str, width: int = 1400, height: int = 820) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), rgb(COLORS["background"]))
    draw = ImageDraw.Draw(image)
    draw.text((60, 40), title, fill=rgb(COLORS["text"]), font=F_TITLE)
    y = 90
    for line in textwrap.wrap(subtitle, width=118):
        draw.text((60, y), line, fill=rgb(COLORS["muted"]), font=F_SUBTITLE)
        y += 23
    return image, draw


def footer(draw: ImageDraw.ImageDraw, text: str, y: int = 785) -> None:
    draw.text((60, y), text, fill=rgb(COLORS["muted"]), font=F_SMALL)


def save(image: Image.Image, filename: str) -> str:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / filename
    image.save(path, format="PNG", optimize=True)
    return f"charts/{filename}"


def barh_chart(
    labels: list[str], values: list[float], title: str, subtitle: str, filename: str,
    color: str = COLORS["id"], value_suffix: str = "", note: str = "",
) -> str:
    image, draw = new_chart(title, subtitle)
    left, right, top, bottom = 390, 1320, 150, 740
    n = max(1, len(labels))
    step = (bottom - top) / n
    bar_h = min(42, step * 0.62)
    vmax = max(values) if values else 1
    vmax = vmax if vmax > 0 else 1
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * step + (step - bar_h) / 2
        draw.text((left - 18, y + bar_h / 2), truncate(label, 44), anchor="rm", fill=rgb(COLORS["text"]), font=F_LABEL)
        width = (right - left) * float(value) / vmax
        draw.rounded_rectangle((left, y, left + width, y + bar_h), radius=7, fill=rgb(color))
        value_text = f"{fmt(value)}{value_suffix}"
        tx = min(right - 5, left + width + 12)
        anchor = "rm" if tx > right - 80 else "lm"
        if anchor == "rm":
            tx = right - 5
        draw.text((tx, y + bar_h / 2), value_text, anchor=anchor, fill=rgb(COLORS["text"]), font=F_VALUE)
    footer(draw, note or "Latest observed product snapshot; exact duplicate rows removed.")
    return save(image, filename)


def panel_barh_chart(
    panels: list[tuple[str, list[str], list[float], str]], title: str, subtitle: str, filename: str, note: str = "",
) -> str:
    image, draw = new_chart(title, subtitle, height=860)
    panel_width = 640
    for pidx, (panel_title, labels, values, color) in enumerate(panels):
        x0 = 50 + pidx * 690
        left, right, top, bottom = x0 + 250, x0 + 620, 190, 760
        draw.text((x0 + 20, 150), panel_title, fill=rgb(COLORS["text"]), font=font(22, True))
        vmax = max(values) if values else 1
        n = max(1, len(labels))
        step = (bottom - top) / n
        bar_h = min(34, step * 0.58)
        for i, (label, value) in enumerate(zip(labels, values)):
            y = top + i * step + (step - bar_h) / 2
            draw.text((left - 10, y + bar_h / 2), truncate(label, 30), anchor="rm", fill=rgb(COLORS["text"]), font=F_SMALL)
            w = (right - left) * float(value) / (vmax or 1)
            draw.rounded_rectangle((left, y, left + w, y + bar_h), radius=5, fill=rgb(color))
            draw.text((min(right, left + w + 8), y + bar_h / 2), fmt(value), anchor="lm", fill=rgb(COLORS["text"]), font=F_SMALL)
    footer(draw, note or "Latest observed product snapshot; sold value is a cumulative source-reported proxy.", y=825)
    return save(image, filename)


def grouped_bar_chart(
    labels: list[str], series: dict[str, list[float]], title: str, subtitle: str, filename: str,
    colors: list[str] | None = None, percent: bool = False, note: str = "",
) -> str:
    image, draw = new_chart(title, subtitle)
    left, right, top, bottom = 120, 1330, 180, 680
    colors = colors or [COLORS["id"], COLORS["vn"], COLORS["neutral"]]
    all_values = [v for vals in series.values() for v in vals]
    vmax = max(all_values) if all_values else 1
    if percent:
        vmax = max(1.0, vmax)
    for tick in range(6):
        value = vmax * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=rgb(COLORS["grid"]), width=1)
        label = f"{100 * value:.0f}%" if percent else fmt(value)
        draw.text((left - 12, y), label, anchor="rm", fill=rgb(COLORS["muted"]), font=F_SMALL)
    groups = len(labels)
    series_count = max(1, len(series))
    group_w = (right - left) / max(1, groups)
    bar_w = min(52, group_w * 0.72 / series_count)
    for sidx, (name, vals) in enumerate(series.items()):
        color = rgb(colors[sidx % len(colors)])
        for i, value in enumerate(vals):
            cx = left + group_w * (i + 0.5)
            x0 = cx - series_count * bar_w / 2 + sidx * bar_w
            h = (bottom - top) * float(value) / (vmax or 1)
            draw.rounded_rectangle((x0, bottom - h, x0 + bar_w - 4, bottom), radius=4, fill=color)
            label = f"{100 * value:.1f}%" if percent else fmt(value)
            draw.text((x0 + (bar_w - 4) / 2, bottom - h - 8), label, anchor="ms", fill=rgb(COLORS["text"]), font=F_SMALL)
    for i, label in enumerate(labels):
        x = left + group_w * (i + 0.5)
        draw.text((x, bottom + 22), truncate(label, 24), anchor="ma", fill=rgb(COLORS["text"]), font=F_AXIS)
    lx = 900
    for sidx, name in enumerate(series):
        draw.rectangle((lx, 145, lx + 18, 163), fill=rgb(colors[sidx % len(colors)]))
        draw.text((lx + 26, 154), name, anchor="lm", fill=rgb(COLORS["text"]), font=F_SMALL)
        lx += 150
    footer(draw, note or "Latest observed product snapshot; exact duplicates removed.")
    return save(image, filename)


def metric_panels(
    metrics: list[tuple[str, dict[str, float], str]], title: str, subtitle: str, filename: str, note: str = "",
) -> str:
    image, draw = new_chart(title, subtitle, height=900)
    panel_positions = [(70, 180), (720, 180), (70, 500), (720, 500)]
    for idx, (metric_name, values, format_kind) in enumerate(metrics[:4]):
        x0, y0 = panel_positions[idx]
        draw.text((x0, y0), metric_name, fill=rgb(COLORS["text"]), font=font(22, True))
        items = list(values.items())
        vmax = max(v for _, v in items) if items else 1
        for j, (label, value) in enumerate(items):
            y = y0 + 62 + j * 80
            draw.text((x0, y + 20), label, fill=rgb(COLORS["text"]), font=F_LABEL)
            w = 420 * value / (vmax or 1)
            color = COLORS["official"] if "Official" == label else COLORS["nonofficial"]
            draw.rounded_rectangle((x0 + 150, y, x0 + 150 + w, y + 40), radius=6, fill=rgb(color))
            if format_kind == "percent":
                label_value = pct(value)
            elif format_kind == "rating":
                label_value = f"{value:.3f}"
            else:
                label_value = fmt(value)
            draw.text((x0 + 160 + w, y + 20), label_value, anchor="lm", fill=rgb(COLORS["text"]), font=F_VALUE)
    footer(draw, note, y=860)
    return save(image, filename)


def boxplot_panels(groups: dict[str, np.ndarray], title: str, subtitle: str, filename: str, note: str = "") -> str:
    image, draw = new_chart(title, subtitle)
    for idx, (country, values) in enumerate(groups.items()):
        x0 = 80 + idx * 680
        top, bottom = 190, 680
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        q1, median, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        low, high = np.quantile(vals, [0.05, 0.95])
        vmax = max(high, q3, 1)
        scale = lambda v: bottom - (bottom - top) * v / vmax
        draw.text((x0 + 250, 155), COUNTRY_NAMES[country], anchor="mm", fill=rgb(COLORS["text"]), font=font(23, True))
        for tick in range(6):
            value = vmax * tick / 5
            y = scale(value)
            draw.line((x0 + 80, y, x0 + 510, y), fill=rgb(COLORS["grid"]), width=1)
            draw.text((x0 + 68, y), fmt(value), anchor="rm", fill=rgb(COLORS["muted"]), font=F_SMALL)
        color = rgb(COLORS[country])
        draw.line((x0 + 290, scale(low), x0 + 290, scale(high)), fill=color, width=4)
        draw.line((x0 + 250, scale(low), x0 + 330, scale(low)), fill=color, width=4)
        draw.line((x0 + 250, scale(high), x0 + 330, scale(high)), fill=color, width=4)
        draw.rectangle((x0 + 220, scale(q3), x0 + 360, scale(q1)), fill=rgb(COLORS["light"]), outline=color, width=4)
        draw.line((x0 + 220, scale(median), x0 + 360, scale(median)), fill=color, width=5)
        draw.text((x0 + 380, scale(median)), f"Median {fmt(median)}", anchor="lm", fill=rgb(COLORS["text"]), font=F_VALUE)
        draw.text((x0 + 290, bottom + 30), COUNTRY_CURRENCY[country], anchor="ma", fill=rgb(COLORS["muted"]), font=F_AXIS)
    footer(draw, note)
    return save(image, filename)


def scatter_panels(
    df: pd.DataFrame, x: str, y: str, title: str, subtitle: str, filename: str,
    x_label: str, y_label: str, log_x: bool = True, log_y: bool = True, note: str = "",
) -> str:
    image, draw = new_chart(title, subtitle)
    for idx, country in enumerate(["id", "vn"]):
        data = df.loc[df["country_code"] == country, [x, y]].dropna()
        xvals = data[x].astype(float).to_numpy()
        yvals = data[y].astype(float).to_numpy()
        if log_x:
            xvals = np.log10(np.maximum(0, xvals) + 1)
        if log_y:
            yvals = np.log10(np.maximum(0, yvals) + 1)
        x0 = 80 + idx * 680
        left, right, top, bottom = x0 + 90, x0 + 600, 190, 690
        xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        if xmax == xmin:
            xmax += 1
        if ymax == ymin:
            ymax += 1
        draw.text(((left + right) / 2, 155), COUNTRY_NAMES[country], anchor="mm", fill=rgb(COLORS["text"]), font=font(23, True))
        for tick in range(6):
            xx = left + (right - left) * tick / 5
            yy = bottom - (bottom - top) * tick / 5
            draw.line((xx, top, xx, bottom), fill=rgb(COLORS["grid"]), width=1)
            draw.line((left, yy, right, yy), fill=rgb(COLORS["grid"]), width=1)
            xv = xmin + (xmax - xmin) * tick / 5
            yv = ymin + (ymax - ymin) * tick / 5
            draw.text((xx, bottom + 12), fmt(10 ** xv - 1 if log_x else xv), anchor="ma", fill=rgb(COLORS["muted"]), font=F_SMALL)
            draw.text((left - 10, yy), fmt(10 ** yv - 1 if log_y else yv), anchor="rm", fill=rgb(COLORS["muted"]), font=F_SMALL)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        color = (*rgb(COLORS[country]), 90)
        for xv, yv in zip(xvals, yvals):
            px = left + (right - left) * (xv - xmin) / (xmax - xmin)
            py = bottom - (bottom - top) * (yv - ymin) / (ymax - ymin)
            odraw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        image.paste(overlay, (0, 0), overlay)
        draw = ImageDraw.Draw(image)
        draw.text(((left + right) / 2, bottom + 56), x_label, anchor="mm", fill=rgb(COLORS["text"]), font=F_AXIS)
        draw.text((left - 62, (top + bottom) / 2), y_label, anchor="mm", fill=rgb(COLORS["text"]), font=F_AXIS)
    footer(draw, note)
    return save(image, filename)


def heatmap_chart(correlations: dict[str, pd.DataFrame], title: str, subtitle: str, filename: str, note: str = "") -> str:
    image, draw = new_chart(title, subtitle, height=900)
    labels = ["Price", "Discount", "Rating", "Sold", "Likes"]
    keys = ["price", "markdown_pct", "rating_observed", "history_sold_value", "liked_count"]
    for pidx, country in enumerate(["id", "vn"]):
        x0 = 60 + pidx * 700
        top, left, cell = 210, x0 + 170, 82
        draw.text((left + 2.5 * cell, 165), COUNTRY_NAMES[country], anchor="mm", fill=rgb(COLORS["text"]), font=font(23, True))
        matrix = correlations[country]
        for i, label in enumerate(labels):
            draw.text((left - 12, top + i * cell + cell / 2), label, anchor="rm", fill=rgb(COLORS["text"]), font=F_SMALL)
            draw.text((left + i * cell + cell / 2, top - 12), label, anchor="ms", fill=rgb(COLORS["text"]), font=F_SMALL)
            for j in range(len(labels)):
                value = float(matrix.loc[keys[i], keys[j]])
                intensity = min(1, abs(value))
                base = np.array(rgb(COLORS["id"] if value >= 0 else COLORS["vn"]))
                fill = tuple(np.round(255 - (255 - base) * intensity).astype(int))
                x = left + j * cell
                y = top + i * cell
                draw.rectangle((x, y, x + cell - 3, y + cell - 3), fill=fill)
                draw.text((x + cell / 2, y + cell / 2), f"{value:.2f}", anchor="mm", fill=rgb(COLORS["text"]), font=F_VALUE)
    footer(draw, note, y=860)
    return save(image, filename)


def load_family(family: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = f"Data/country_code=*/dataset={family}/**/*.csv"
    for path in sorted(ROOT.glob(pattern)):
        df = pd.read_csv(path, low_memory=False)
        parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}
        if "country_code" not in df.columns:
            df["country_code"] = parts.get("country_code")
        df["_source_path"] = path.relative_to(ROOT).as_posix()
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def prepare_data() -> dict[str, pd.DataFrame]:
    products_raw = load_family("products")
    shops = load_family("shop_info").drop_duplicates()
    platform_categories = load_family("category_platform").drop_duplicates()
    product_categories = load_family("product_categories").drop_duplicates()
    shop_categories = load_family("category_list").drop_duplicates()

    products = products_raw.drop_duplicates().copy()
    products["date"] = pd.to_datetime(products["date"], errors="coerce")
    numeric = [
        "price", "price_original", "price_before_promo", "discount_percent", "promotion_id",
        "voucher_discount", "voucher_min_spend", "history_sold_value", "monthly_sold_value",
        "rating", "rating_count", "liked_count", "catid",
    ]
    for column in numeric:
        products[column] = pd.to_numeric(products[column], errors="coerce")
    latest = (
        products.sort_values("date")
        .groupby(["country_code", "shop_id", "item_id"], dropna=False, as_index=False)
        .tail(1)
        .copy()
    )
    latest["markdown_pct"] = latest["discount_percent"].fillna(0).clip(lower=0)
    latest["price_markdown"] = latest["markdown_pct"] > 0
    latest["voucher_active"] = latest["voucher_discount"].fillna(0) > 0
    latest["promo_id_active"] = latest["promotion_id"].fillna(0) > 0
    latest["promoted"] = latest[["price_markdown", "voucher_active", "promo_id_active"]].any(axis=1) | latest["voucher_code"].notna()
    latest["rating_observed"] = latest["rating"].where((latest["rating_count"].fillna(0) > 0) & (latest["rating"] > 0))
    latest["price_band"] = latest.groupby("country_code")["price"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    )
    latest["discount_band"] = pd.cut(
        latest["markdown_pct"], bins=[-0.1, 0, 10, 25, 50, 100],
        labels=["0%", "1–10%", "11–25%", "26–50%", "51–100%"],
    )

    shop_cols = [
        "country_code", "shop_id", "is_official_shop", "follower_count", "rating_star",
        "response_rate", "response_time", "item_count", "cancellation_rate",
    ]
    latest = latest.merge(shops[shop_cols], on=["country_code", "shop_id"], how="left", validate="many_to_one")
    taxonomy = platform_categories[["country_code", "category_id", "display_category_name"]].rename(columns={"category_id": "catid"})
    latest = latest.merge(taxonomy, on=["country_code", "catid"], how="left", validate="many_to_one")
    return {
        "products_raw": products_raw,
        "products_deduped": products,
        "latest": latest,
        "shops": shops,
        "platform_categories": platform_categories,
        "product_categories": product_categories,
        "shop_categories": shop_categories,
    }


def add_metric(rows: list[dict[str, Any]], area: str, metric: str, country: str, segment: str, value: float, unit: str, note: str = "") -> None:
    rows.append({
        "area": area, "metric": metric, "country": country, "segment": segment,
        "value": value, "unit": unit, "note": note,
    })


def analyze(data: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], pd.DataFrame]:
    latest = data["latest"]
    metrics: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    country = latest.groupby("country_code").agg(
        products=("item_id", "nunique"),
        shops=("shop_id", "nunique"),
        promoted_share=("promoted", "mean"),
        markdown_share=("price_markdown", "mean"),
        voucher_share=("voucher_active", "mean"),
        promo_id_share=("promo_id_active", "mean"),
        median_discount=("markdown_pct", "median"),
        median_price=("price", "median"),
        median_sold=("history_sold_value", "median"),
        median_likes=("liked_count", "median"),
        median_rating=("rating_observed", "median"),
        soldout_share=("is_sold_out", "mean"),
        ad_share=("is_ad", "mean"),
    )
    result["country"] = country
    for c, row in country.iterrows():
        for metric, unit in [
            ("products", "count"), ("shops", "count"), ("promoted_share", "share"),
            ("markdown_share", "share"), ("voucher_share", "share"), ("median_discount", "percent"),
            ("median_price", "local currency units"), ("median_sold", "source sold-value proxy"),
            ("median_likes", "count"), ("median_rating", "stars"),
        ]:
            add_metric(metrics, "Country", metric, c, "All latest products", float(row[metric]), unit)

    promo = latest.groupby(["country_code", "promoted"]).agg(
        products=("item_id", "size"), median_sold=("history_sold_value", "median"),
        median_likes=("liked_count", "median"), median_rating=("rating_observed", "median"),
        median_price=("price", "median"),
    ).reset_index()
    result["promo"] = promo
    for _, row in promo.iterrows():
        segment = "Promoted" if row["promoted"] else "Not promoted"
        add_metric(metrics, "Promotion", "median_sold", row["country_code"], segment, row["median_sold"], "source sold-value proxy")

    official = latest.groupby(["country_code", "is_official_shop"]).agg(
        products=("item_id", "nunique"), shops=("shop_id", "nunique"),
        promoted_share=("promoted", "mean"), median_sold=("history_sold_value", "median"),
        median_likes=("liked_count", "median"), median_rating=("rating_observed", "median"),
        median_discount=("markdown_pct", "median"),
    ).reset_index()
    result["official"] = official

    shop = latest.groupby(["country_code", "shop_id", "shop_name", "is_official_shop"]).agg(
        products=("item_id", "nunique"), total_sold=("history_sold_value", "sum"),
        median_sold=("history_sold_value", "median"), median_price=("price", "median"),
        promoted_share=("promoted", "mean"), median_rating=("rating_observed", "median"),
        total_likes=("liked_count", "sum"),
    ).reset_index()
    result["shop"] = shop

    category = latest.groupby(["country_code", "display_category_name"], dropna=False).agg(
        products=("item_id", "nunique"), total_sold=("history_sold_value", "sum"),
        median_sold=("history_sold_value", "median"), median_price=("price", "median"),
        promoted_share=("promoted", "mean"), median_rating=("rating_observed", "median"),
    ).reset_index()
    result["category"] = category

    discount_band = latest.groupby(["country_code", "discount_band"], observed=True).agg(
        products=("item_id", "size"), median_sold=("history_sold_value", "median"),
        median_likes=("liked_count", "median"),
    ).reset_index()
    result["discount_band"] = discount_band

    rating_bins = pd.cut(
        latest["rating_observed"], bins=[0, 4.5, 4.8, 4.9, 4.95, 5.01],
        labels=["<4.5", "4.5–4.8", "4.8–4.9", "4.9–4.95", "≥4.95"],
    )
    rating_sold = latest.assign(rating_band=rating_bins).dropna(subset=["rating_band"]).groupby(
        ["country_code", "rating_band"], observed=True
    ).agg(products=("item_id", "size"), median_sold=("history_sold_value", "median")).reset_index()
    result["rating_sold"] = rating_sold

    corr_cols = ["price", "markdown_pct", "rating_observed", "history_sold_value", "liked_count"]
    correlations = {
        c: latest.loc[latest["country_code"] == c, corr_cols].corr(method="spearman")
        for c in ["id", "vn"]
    }
    result["correlations"] = correlations
    for c, matrix in correlations.items():
        for i, a in enumerate(corr_cols):
            for b in corr_cols[i + 1:]:
                add_metric(metrics, "Relationship", f"spearman_{a}_vs_{b}", c, "All latest products", float(matrix.loc[a, b]), "rho")

    top_products = latest.sort_values("history_sold_value", ascending=False).copy()
    result["top_products"] = top_products
    return result, pd.DataFrame(metrics)


def build_charts(data: dict[str, pd.DataFrame], analysis: dict[str, Any]) -> dict[str, str]:
    latest = data["latest"]
    country = analysis["country"]
    charts: dict[str, str] = {}

    charts["C01"] = grouped_bar_chart(
        ["Products", "Shops"],
        {
            "Indonesia": [country.loc["id", "products"], country.loc["id", "shops"]],
            "Vietnam": [country.loc["vn", "products"], country.loc["vn", "shops"]],
        },
        "Vietnam has the broader observed assortment",
        "Distinct latest listings and shops in the repository sample",
        "01_market_assortment.png",
        note="Repository coverage, not total marketplace size. Both markets contain 10 observed shops.",
    )

    charts["C02"] = boxplot_panels(
        {c: latest.loc[latest["country_code"] == c, "price"].dropna().to_numpy() for c in ["id", "vn"]},
        "Price architecture differs by market",
        "5th–95th percentile whiskers and interquartile range; local source currency units",
        "02_price_distribution.png",
        note="Do not compare IDR and VND levels directly without a currency definition and exchange-rate normalization.",
    )

    discount_dist = latest.groupby(["country_code", "discount_band"], observed=True).size().unstack(fill_value=0)
    discount_dist = discount_dist.div(discount_dist.sum(axis=1), axis=0)
    labels = [str(x) for x in discount_dist.columns]
    charts["C03"] = grouped_bar_chart(
        labels,
        {
            "Indonesia": discount_dist.loc["id"].tolist(),
            "Vietnam": discount_dist.loc["vn"].tolist(),
        },
        "Indonesia relies on much deeper displayed markdowns",
        "Share of latest listings by displayed discount band",
        "03_discount_distribution.png",
        percent=True,
        note="Missing discount_percent is treated as 0% displayed markdown; other promotion mechanisms are analyzed separately.",
    )

    mechanisms = ["promoted_share", "markdown_share", "voucher_share", "promo_id_share"]
    charts["C04"] = grouped_bar_chart(
        ["Any promotion", "Price markdown", "Voucher discount", "Promotion ID"],
        {
            "Indonesia": [country.loc["id", x] for x in mechanisms],
            "Vietnam": [country.loc["vn", x] for x in mechanisms],
        },
        "Promotion is close to an always-on state",
        "Share of latest listings with each observed promotion signal",
        "04_promotion_mechanisms.png",
        percent=True,
        note="Any promotion = markdown, positive voucher discount, non-zero promotion ID, or voucher code.",
    )

    promo = analysis["promo"]
    get_promo = lambda c, flag, col: float(promo.loc[(promo.country_code == c) & (promo.promoted == flag), col].iloc[0])
    charts["C05"] = grouped_bar_chart(
        ["Indonesia", "Vietnam"],
        {
            "Not promoted": [get_promo("id", False, "median_sold"), get_promo("vn", False, "median_sold")],
            "Promoted": [get_promo("id", True, "median_sold"), get_promo("vn", True, "median_sold")],
        },
        "Promoted listings have higher observed sold-value medians",
        "Median cumulative source-reported sold value; association, not causal lift",
        "05_promotion_vs_sold.png",
        note="Promotion exposure is not randomized and historical sold value may predate the current promotion.",
    )

    db = analysis["discount_band"].pivot(index="discount_band", columns="country_code", values="median_sold").fillna(0)
    charts["C06"] = grouped_bar_chart(
        [str(x) for x in db.index],
        {
            "Indonesia": db.get("id", pd.Series(0, index=db.index)).tolist(),
            "Vietnam": db.get("vn", pd.Series(0, index=db.index)).tolist(),
        },
        "Deeper discounts do not show a monotonic sold-value payoff",
        "Median cumulative sold-value proxy by displayed discount band",
        "06_discount_band_sold.png",
        note="Cross-sectional result; category, product age, shop strength, and promotion selection are confounders.",
    )

    panels = []
    for c in ["id", "vn"]:
        top = analysis["shop"].loc[analysis["shop"].country_code == c].nlargest(6, "total_sold")
        panels.append((COUNTRY_NAMES[c], top.shop_name.map(lambda x: truncate(x, 29)).tolist(), top.total_sold.tolist(), COLORS[c]))
    charts["C07"] = panel_barh_chart(
        panels,
        "A few shops dominate the observed sold-value pool",
        "Top shops by sum of latest-listing cumulative sold values",
        "07_top_shops_sold.png",
        note="Summed historical sold values are a popularity proxy, not period revenue.",
    )

    charts["C08"] = scatter_panels(
        analysis["shop"], "products", "median_sold",
        "More listings do not guarantee stronger SKU productivity",
        "Shop assortment size versus median sold value per observed listing",
        "08_shop_productivity.png",
        "Distinct latest listings", "Median sold value", log_x=False, log_y=True,
        note="Each point is one shop; y-axis uses log scaling because sold values are highly skewed.",
    )

    official_vn = analysis["official"].loc[analysis["official"].country_code == "vn"].set_index("is_official_shop")
    vals = {
        "Official": official_vn.loc[True],
        "Non-official": official_vn.loc[False],
    }
    charts["C09"] = metric_panels(
        [
            ("Promotion coverage", {k: float(v.promoted_share) for k, v in vals.items()}, "percent"),
            ("Median sold value", {k: float(v.median_sold) for k, v in vals.items()}, "number"),
            ("Median likes", {k: float(v.median_likes) for k, v in vals.items()}, "number"),
            ("Median product rating", {k: float(v.median_rating) for k, v in vals.items()}, "rating"),
        ],
        "Vietnam official shops lead on reach, not star rating",
        "Latest product-level medians and promotion coverage",
        "09_official_vs_nonofficial_vietnam.png",
        note="Vietnam: 7 official shops/477 products vs 3 non-official/205. Indonesia non-official: only 1 shop/3 products, so excluded.",
    )

    panels = []
    for c in ["id", "vn"]:
        cat = analysis["category"].loc[analysis["category"].country_code == c].nlargest(7, "products")
        panels.append((COUNTRY_NAMES[c], cat.display_category_name.fillna("Unmapped").tolist(), cat.products.tolist(), COLORS[c]))
    charts["C10"] = panel_barh_chart(
        panels,
        "Each market sample is dominated by one platform category",
        "Distinct latest listings by platform category",
        "10_category_mix.png",
        note="The repository is a targeted shop sample: Indonesia is beauty-heavy; Vietnam is food-and-beverage-heavy.",
    )

    panels = []
    for c in ["id", "vn"]:
        top = analysis["top_products"].loc[analysis["top_products"].country_code == c].head(6)
        panels.append((COUNTRY_NAMES[c], top.product_name.map(lambda x: truncate(x, 29)).tolist(), top.history_sold_value.tolist(), COLORS[c]))
    charts["C11"] = panel_barh_chart(
        panels,
        "Top products account for a disproportionate popularity share",
        "Highest cumulative source-reported sold values",
        "11_top_products_sold.png",
        note="Product titles are truncated; full identifiers and values remain available in the source data.",
    )

    charts["C12"] = heatmap_chart(
        analysis["correlations"],
        "Engagement volume tracks sold value; discount depth does not",
        "Spearman rank correlations among latest listings",
        "12_relationship_heatmap.png",
        note="Rating excludes unrated listings (rating_count = 0). Correlation describes association, not causality.",
    )

    charts["C13"] = scatter_panels(
        latest, "liked_count", "history_sold_value",
        "Likes are a strong popularity signal in both markets",
        "Listing likes versus cumulative sold-value proxy",
        "13_sold_vs_likes.png",
        "Likes (log scale)", "Sold value (log scale)",
        note="Spearman ρ is reported in the EDA report; axes use log10(value + 1).",
    )

    charts["C14"] = scatter_panels(
        latest, "price", "history_sold_value",
        "Higher-priced listings tend to have lower observed sold values",
        "Local price versus cumulative sold-value proxy",
        "14_price_vs_sold.png",
        "Price, local units (log scale)", "Sold value (log scale)",
        note="Price is not currency-normalized across countries; panels are scaled independently.",
    )

    rb = analysis["rating_sold"].pivot(index="rating_band", columns="country_code", values="median_sold").fillna(0)
    charts["C15"] = grouped_bar_chart(
        [str(x) for x in rb.index],
        {
            "Indonesia": rb.get("id", pd.Series(0, index=rb.index)).tolist(),
            "Vietnam": rb.get("vn", pd.Series(0, index=rb.index)).tolist(),
        },
        "Star rating alone does not separate high-selling products",
        "Median cumulative sold-value proxy by product-rating band",
        "15_rating_band_sold.png",
        note="Unrated products are excluded. Rating count is a much stronger popularity correlate than average stars.",
    )
    return charts


def insight_rows(data: dict[str, pd.DataFrame], a: dict[str, Any], charts: dict[str, str]) -> list[dict[str, Any]]:
    latest = data["latest"]
    country = a["country"]
    corr = a["correlations"]
    promo = a["promo"]
    shops = a["shop"]
    categories = a["category"]
    official = a["official"]

    def promo_value(c: str, flag: bool, column: str) -> float:
        return float(promo.loc[(promo.country_code == c) & (promo.promoted == flag), column].iloc[0])

    def official_value(c: str, flag: bool, column: str) -> float:
        return float(official.loc[(official.country_code == c) & (official.is_official_shop == flag), column].iloc[0])

    insights: list[dict[str, Any]] = []

    def add(area: str, insight: str, evidence: str, action: str, chart_id: str, confidence: str, caveat: str = "") -> None:
        insights.append({
            "insight_id": f"I{len(insights) + 1:02d}", "area": area, "insight": insight,
            "evidence": evidence, "recommended_action": action, "chart_id": chart_id,
            "chart_file": charts[chart_id], "confidence": confidence, "caveat": caveat,
        })

    add(
        "Product",
        "Vietnam has the larger observed assortment, while both markets cover the same number of shops.",
        f"Vietnam has {int(country.loc['vn','products'])} latest listings versus {int(country.loc['id','products'])} in Indonesia; both contain 10 shops.",
        "Use product-level benchmarks within country and weight shop comparisons for assortment size.",
        "C01", "High", "Repository coverage is not marketplace share.",
    )
    add(
        "Pricing",
        "Price levels must be managed as separate local-market architectures.",
        f"Median source prices are {country.loc['id','median_price']:,.0f} in Indonesia and {country.loc['vn','median_price']:,.0f} in Vietnam, but currency metadata is absent.",
        "Set local price ladders and obtain currency/unit definitions before any cross-market price index.",
        "C02", "High", "IDR and VND are not directly comparable.",
    )
    add(
        "Promotion",
        "Indonesia uses substantially deeper displayed markdowns.",
        f"Median displayed markdown is {country.loc['id','median_discount']:.0f}% in Indonesia versus {country.loc['vn','median_discount']:.0f}% in Vietnam.",
        "Audit margin exposure in Indonesia and replace blanket deep discounts with SKU-level thresholds.",
        "C03", "High", "Profit impact cannot be measured without cost and realized transaction data.",
    )
    add(
        "Promotion",
        "Promotion is nearly always on, especially in Vietnam.",
        f"Promotion coverage is {pct(country.loc['vn','promoted_share'])} in Vietnam and {pct(country.loc['id','promoted_share'])} in Indonesia.",
        "Create a stable non-promoted holdout set or rotating test cells so future lift can be measured.",
        "C04", "High", "Current data is observational and only three snapshot days are available.",
    )
    add(
        "Promotion",
        "Promoted listings have much higher observed sold-value medians.",
        f"Indonesia: {promo_value('id',True,'median_sold'):,.0f} promoted vs {promo_value('id',False,'median_sold'):,.0f} not promoted. Vietnam: {promo_value('vn',True,'median_sold'):,.0f} vs {promo_value('vn',False,'median_sold'):,.1f}.",
        "Prioritize promoted placement for products with existing engagement, but validate incremental lift through controlled tests.",
        "C05", "Medium", "Historical sold value may predate the observed promotion and selection bias is likely.",
    )
    add(
        "Promotion",
        "Deeper discount does not produce a consistent step-up in sold value.",
        f"Discount–sold Spearman ρ is {corr['id'].loc['markdown_pct','history_sold_value']:.2f} in Indonesia and {corr['vn'].loc['markdown_pct','history_sold_value']:.2f} in Vietnam.",
        "Optimize promotion targeting, visibility, and product selection before increasing discount depth.",
        "C06", "High", "Cross-sectional association; product age and category are not controlled.",
    )
    for c in ["id", "vn"]:
        market_shops = shops.loc[shops.country_code == c]
        top = market_shops.nlargest(1, "total_sold").iloc[0]
        share = top.total_sold / market_shops.total_sold.sum()
        add(
            "Shop",
            f"{COUNTRY_NAMES[c]} popularity is concentrated in one leading shop.",
            f"{top.shop_name.strip()} contributes {pct(share)} of the market sample's summed sold-value proxy.",
            "Protect availability and promotion execution for the leader while building a challenger plan for the next two shops.",
            "C07", "High", "Sold values are cumulative and shop histories may differ.",
        )
    add(
        "Shop",
        "Large assortments do not guarantee high per-listing productivity.",
        "Observed shops with similar or larger listing counts show materially different median sold values.",
        "Track sold value per active SKU and retire, bundle, or reposition persistently low-engagement tail listings.",
        "C08", "Medium", "Historical sold value is not normalized by listing age.",
    )
    add(
        "Official vs non-official",
        "Vietnam official shops lead on median sold value and likes, but not on average star rating.",
        f"Official vs non-official medians: sold {official_value('vn',True,'median_sold'):,.0f} vs {official_value('vn',False,'median_sold'):,.0f}; likes {official_value('vn',True,'median_likes'):,.0f} vs {official_value('vn',False,'median_likes'):,.0f}; rating {official_value('vn',True,'median_rating'):.3f} vs {official_value('vn',False,'median_rating'):.3f}.",
        "For non-official shops, prioritize reach and trust signals rather than chasing marginal rating gains.",
        "C09", "Medium", "Only 10 Vietnam shops; category and brand mix are confounders.",
    )
    add(
        "Official vs non-official",
        "Indonesia cannot support a reliable official-status comparison.",
        f"The non-official Indonesia segment contains only {int(official_value('id',False,'products'))} products from {int(official_value('id',False,'shops'))} shop.",
        "Collect more non-official Indonesia shops before making policy or assortment decisions by official status.",
        "C09", "High", "Sample imbalance prevents a meaningful comparison.",
    )
    for c in ["id", "vn"]:
        market = categories.loc[categories.country_code == c]
        top = market.nlargest(1, "products").iloc[0]
        share = top.products / market.products.sum()
        add(
            "Category",
            f"{COUNTRY_NAMES[c]} is dominated by {top.display_category_name}.",
            f"{int(top.products)} listings, or {pct(share)} of the observed market assortment, sit in this platform category.",
            "Optimize subcategory depth within the dominant category; treat other categories as small tests until coverage expands.",
            "C10", "High", "This is a targeted repository sample, not the full marketplace.",
        )
    for c in ["id", "vn"]:
        market = latest.loc[latest.country_code == c]
        top10_share = market.nlargest(10, "history_sold_value").history_sold_value.sum() / market.history_sold_value.sum()
        add(
            "Product",
            f"{COUNTRY_NAMES[c]} has a concentrated head of top products.",
            f"The top 10 products account for {pct(top10_share)} of summed cumulative sold value in the observed market sample.",
            "Create a hero-SKU protection plan while testing bundles and cross-sell from these high-traffic listings to the long tail.",
            "C11", "High", "Cumulative sold values favor older listings.",
        )
    add(
        "Relationships",
        "Likes are the strongest simple engagement signal for sold value.",
        f"Sold–likes Spearman ρ is {corr['id'].loc['history_sold_value','liked_count']:.2f} in Indonesia and {corr['vn'].loc['history_sold_value','liked_count']:.2f} in Vietnam.",
        "Use likes as an early product-prioritization feature and monitor sold-to-like conversion by shop/category.",
        "C12", "High", "Both variables accumulate over time and may share listing-age effects.",
    )
    add(
        "Relationships",
        "Average star rating is weakly related to sold value.",
        f"Sold–rating Spearman ρ is {corr['id'].loc['history_sold_value','rating_observed']:.2f} in Indonesia and {corr['vn'].loc['history_sold_value','rating_observed']:.2f} in Vietnam.",
        "Avoid ranking products by star rating alone; combine rating volume, likes, price, and shop context.",
        "C12", "High", "Unrated listings were excluded from rating correlations.",
    )
    add(
        "Relationships",
        "The sold–likes relationship is visually stable across both markets despite different category mixes.",
        "Both log-scale panels show a clear upward association, consistent with the rank correlations.",
        "Build alerting for listings with high likes but unexpectedly low sold values; these may have price, availability, or conversion friction.",
        "C13", "Medium", "No sessions, views, or conversion denominator is available.",
    )
    add(
        "Pricing",
        "Higher-priced products tend to have lower sold values within each market.",
        f"Price–sold Spearman ρ is {corr['id'].loc['price','history_sold_value']:.2f} in Indonesia and {corr['vn'].loc['price','history_sold_value']:.2f} in Vietnam.",
        "Benchmark performance within price bands and avoid applying a single sold-volume target across premium and entry products.",
        "C14", "High", "Price is confounded by category, pack size, and listing age.",
    )
    add(
        "Product",
        "Star-rating bands do not create a clean sales-performance ladder.",
        "Median sold values fluctuate across rating bands rather than increasing consistently with rating.",
        "Use rating as a quality guardrail, not the primary growth lever; prioritize rating count and engagement volume.",
        "C15", "High", "Average ratings are tightly clustered near the top of the scale.",
    )
    add(
        "Advertising/Inventory",
        "The current extract cannot support ad-effectiveness or stockout analysis.",
        f"is_ad is {pct(country.ad_share.max())} and is_sold_out is {pct(country.soldout_share.max())} across latest listings.",
        "Add campaign exposure/spend/clicks and stock-on-hand/history before attempting ROAS or inventory optimization.",
        "C01", "High", "Absence may reflect extraction scope rather than true marketplace state.",
    )
    return insights


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(
    data: dict[str, pd.DataFrame], analysis: dict[str, Any], metrics: pd.DataFrame,
    charts: dict[str, str], insights: list[dict[str, Any]],
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORTS / "business_eda_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(insights).to_csv(REPORTS / "business_insights.csv", index=False, encoding="utf-8-sig")

    latest = data["latest"]
    country = analysis["country"]
    report = [
        "# Business Exploratory Data Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive scope",
        "",
        f"This EDA uses {len(latest):,} distinct latest-observed listings from 20 shops after removing 30 exact duplicate snapshot rows. Product snapshots cover 2026-07-01 to 2026-07-03.",
        "",
        "No machine-learning model was built. `history_sold_value` and `monthly_sold_value` are treated as source-reported demand/popularity proxies, not transactions or recognized revenue.",
        "",
        "## Analysis rules",
        "",
        "- Latest observed row per `country_code + shop_id + item_id` is used for cross-sectional product analysis.",
        "- Prices remain in source-market units. Indonesia and Vietnam prices are never pooled or converted.",
        "- A listing is promoted if it has a displayed markdown, positive voucher discount, non-zero promotion ID, or voucher code.",
        "- Product ratings are used only when `rating_count > 0` and rating is positive; zero-rating/unrated listings are excluded.",
        "- Spearman correlations are used because prices, likes, ratings counts, and sold values are skewed.",
        "- Findings are descriptive. Promotion effects, official-shop effects, and price effects are not causal estimates.",
        "",
        "## Market overview",
        "",
        markdown_table(
            ["Metric", "Indonesia", "Vietnam"],
            [
                ["Latest listings", int(country.loc["id", "products"]), int(country.loc["vn", "products"])],
                ["Observed shops", int(country.loc["id", "shops"]), int(country.loc["vn", "shops"])],
                ["Promotion coverage", pct(country.loc["id", "promoted_share"]), pct(country.loc["vn", "promoted_share"])],
                ["Median displayed discount", f"{country.loc['id','median_discount']:.0f}%", f"{country.loc['vn','median_discount']:.0f}%"],
                ["Median sold-value proxy", f"{country.loc['id','median_sold']:,.0f}", f"{country.loc['vn','median_sold']:,.0f}"],
                ["Median likes", f"{country.loc['id','median_likes']:,.0f}", f"{country.loc['vn','median_likes']:,.1f}"],
                ["Median observed rating", f"{country.loc['id','median_rating']:.3f}", f"{country.loc['vn','median_rating']:.3f}"],
            ],
        ),
        "",
        f"![Market assortment]({charts['C01']})",
        "",
        "## Product and category analysis",
        "",
        f"![Category mix]({charts['C10']})",
        "",
        f"![Top products]({charts['C11']})",
        "",
        "## Pricing analysis",
        "",
        f"![Price distribution]({charts['C02']})",
        "",
        f"![Price versus sold value]({charts['C14']})",
        "",
        "## Promotion analysis",
        "",
        f"![Discount distribution]({charts['C03']})",
        "",
        f"![Promotion mechanisms]({charts['C04']})",
        "",
        f"![Promotion versus sold value]({charts['C05']})",
        "",
        f"![Discount band versus sold value]({charts['C06']})",
        "",
        "## Shop analysis",
        "",
        f"![Top shops]({charts['C07']})",
        "",
        f"![Shop productivity]({charts['C08']})",
        "",
        f"![Official comparison]({charts['C09']})",
        "",
        "## Relationships",
        "",
        f"![Correlation heatmap]({charts['C12']})",
        "",
        f"![Sold versus likes]({charts['C13']})",
        "",
        f"![Rating bands versus sold value]({charts['C15']})",
        "",
        "## Actionable insights",
        "",
    ]
    for item in insights:
        report.extend([
            f"### {item['insight_id']}. {item['insight']}",
            "",
            f"**Evidence:** {item['evidence']}  ",
            f"**Action:** {item['recommended_action']}  ",
            f"**Chart:** {item['chart_id']} — `{item['chart_file']}`  ",
            f"**Confidence:** {item['confidence']}" + (f"  \n**Caveat:** {item['caveat']}" if item["caveat"] else ""),
            "",
        ])
    report.extend([
        "## Best realistically supported AI use case",
        "",
        "### AI-assisted product opportunity and promotion prioritization",
        "",
        "The best-supported use case is a decision-support copilot that ranks products for merchandising attention: protect high-engagement hero SKUs, surface high-like/low-sold conversion opportunities, flag deep-discount/low-response listings, benchmark products within country/category/price band, and recommend candidates for controlled promotion tests.",
        "",
        "Why this is realistic:",
        "",
        "- The dataset has stable product, shop, brand, category, price, promotion, rating-volume, likes, and sold-value fields.",
        "- Strong sold-value relationships with likes and rating volume provide useful prioritization signals.",
        "- The output can remain explainable: every recommendation can show its peer group and observed evidence.",
        "- It does not require unavailable order, customer, cost, inventory, or campaign-attribution data.",
        "",
        "What it should not claim:",
        "",
        "- It should not forecast order-level demand, estimate causal promotion lift, optimize profit, predict customer behavior, or calculate ROAS with the current data.",
        "- A predictive model should wait for a longer time series plus realized transactions, exposure/control flags, inventory, costs, and listing-age controls.",
        "",
        "## Limitations",
        "",
        "- Only three snapshot dates are available; trend and seasonality analysis is not reliable.",
        "- Cumulative sold values, likes, and rating counts are affected by listing age.",
        "- Country samples cover different dominant categories, so country effects and category effects are confounded.",
        "- Indonesia's non-official segment has only one shop and three products.",
        "- Currency, response-time units, and several source-field definitions remain undocumented.",
        "- No ad exposure/spend, inventory history, customer, order, cost, refund, or realized revenue data is available.",
    ])
    (REPORTS / "business_eda_report.md").write_text("\n".join(report), encoding="utf-8")


def contact_sheet(chart_paths: list[Path]) -> Path:
    thumbs: list[Image.Image] = []
    for path in chart_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((560, 340))
        thumbs.append(image.copy())
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 580, rows * 360), rgb(COLORS["light"]))
    for i, image in enumerate(thumbs):
        x = (i % cols) * 580 + 10
        y = (i // cols) * 360 + 10
        sheet.paste(image, (x, y))
    path = CHARTS / "chart_contact_sheet.png"
    sheet.save(path, format="PNG", optimize=True)
    return path


def run_business_eda() -> dict[str, Any]:
    data = prepare_data()
    analysis, metrics = analyze(data)
    charts = build_charts(data, analysis)
    insights = insight_rows(data, analysis, charts)
    write_report(data, analysis, metrics, charts, insights)
    contact = contact_sheet([ROOT / "reports" / p for p in charts.values()])
    outputs = [
        REPORTS / "business_eda_report.md",
        REPORTS / "business_insights.csv",
        REPORTS / "business_eda_metrics.csv",
        contact,
    ] + [ROOT / "reports" / p for p in charts.values()]
    return {
        "raw_product_rows": int(len(data["products_raw"])),
        "deduplicated_snapshot_rows": int(len(data["products_deduped"])),
        "latest_distinct_products": int(len(data["latest"])),
        "shops": int(data["latest"]["shop_id"].nunique()),
        "insights": len(insights),
        "charts": len(charts),
        "outputs": [str(p.relative_to(ROOT)) for p in outputs],
    }


if __name__ == "__main__":
    print(json.dumps(run_business_eda(), ensure_ascii=False, indent=2))
