from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
PRODUCT_KEY = ["country_code", "shop_id", "item_id", "date"]


def load_family(root: Path, family: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"Data/country_code=*/dataset={family}/**/*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        if "country_code" not in frame:
            frame["country_code"] = parts.get("country_code")
        frame["_source_path"] = path.relative_to(root).as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No files found for dataset={family}")
    return pd.concat(frames, ignore_index=True, sort=False)


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _assert_many_to_one(frame: pd.DataFrame, key: list[str], name: str) -> None:
    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"{name} is not unique on {key}; duplicate rows={duplicates}")


def _join_rate(frame: pd.DataFrame, marker: str) -> dict[str, Any]:
    matched = int(frame[marker].notna().sum())
    return {
        "rows": int(len(frame)), "matched": matched, "unmatched": int(len(frame) - matched),
        "match_rate": matched / len(frame) if len(frame) else 0.0,
    }


def build_latest_product_table(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    products_raw = load_family(root, "products")
    shops = load_family(root, "shop_info").drop_duplicates()
    platform = load_family(root, "category_platform").drop_duplicates()
    product_categories = load_family(root, "product_categories").drop_duplicates()
    shop_categories = load_family(root, "category_list").drop_duplicates()

    required_product = [
        "country_code", "shop_id", "item_id", "date", "price", "price_original",
        "price_before_promo", "discount_percent", "history_sold_value", "monthly_sold_value",
        "liked_count", "rating", "rating_count", "catid",
    ]
    require_columns(products_raw, required_product, "products")

    duplicate_rows = int(products_raw.duplicated().sum())
    known_file = products_raw["_source_path"].str.contains(
        r"country_code=vn/dataset=products/shop_id=289646907/products.csv", regex=False
    )
    known_duplicates = int(products_raw.loc[known_file].duplicated().sum())
    if known_duplicates != 30:
        raise ValueError(f"Known duplicate validation failed: expected 30, found {known_duplicates}")

    products = products_raw.drop_duplicates().copy()
    products["date"] = pd.to_datetime(products["date"], errors="raise")
    numeric = [
        "price", "price_original", "price_before_promo", "discount_percent", "promotion_id",
        "voucher_discount", "voucher_min_spend", "history_sold_value", "monthly_sold_value",
        "liked_count", "rating", "rating_count", "catid",
    ]
    for column in numeric:
        products[column] = pd.to_numeric(products[column], errors="coerce")
    latest = products.sort_values("date").groupby(
        ["country_code", "shop_id", "item_id"], dropna=False, as_index=False
    ).tail(1).copy()
    if latest.duplicated(["country_code", "shop_id", "item_id"]).any():
        raise ValueError("Latest selection did not produce one row per country/shop/item")

    shop_cols = [
        "country_code", "shop_id", "shop_name", "rating_star", "follower_count", "item_count",
        "is_official_shop", "response_rate", "cancellation_rate",
    ]
    _assert_many_to_one(shops[shop_cols], ["country_code", "shop_id"], "shop_info")
    latest = latest.merge(
        shops[shop_cols].rename(columns={"shop_name": "shop_name_info"}),
        on=["country_code", "shop_id"], how="left", validate="many_to_one", indicator="_shop_join",
    )
    shop_join = {
        "rows": len(latest), "matched": int((latest["_shop_join"] == "both").sum()),
        "unmatched": int((latest["_shop_join"] != "both").sum()),
    }
    shop_join["match_rate"] = shop_join["matched"] / len(latest)
    latest.drop(columns="_shop_join", inplace=True)
    latest["shop_name"] = latest["shop_name_info"].fillna(latest.get("shop_name"))
    latest.drop(columns=["shop_name_info"], inplace=True)

    taxonomy = platform[["country_code", "category_id", "display_category_name"]].rename(
        columns={"category_id": "catid", "display_category_name": "platform_category"}
    )
    _assert_many_to_one(taxonomy, ["country_code", "catid"], "category_platform")
    latest = latest.merge(taxonomy, on=["country_code", "catid"], how="left", validate="many_to_one")
    platform_join = _join_rate(latest, "platform_category")

    pc = product_categories.copy()
    pc["date"] = pd.to_datetime(pc["date"], errors="raise")
    sc = shop_categories[["country_code", "shop_id", "shop_category_id", "date", "display_name", "total"]].copy()
    sc["date"] = pd.to_datetime(sc["date"], errors="raise")
    _assert_many_to_one(sc, ["country_code", "shop_id", "shop_category_id", "date"], "category_list")
    pc = pc.merge(
        sc, left_on=["country_code", "shop_id", "category_id", "date"],
        right_on=["country_code", "shop_id", "shop_category_id", "date"],
        how="left", validate="many_to_one",
    )
    product_category_child_match = float(pc["display_name"].notna().mean())
    pc["_total"] = pd.to_numeric(pc["total"], errors="coerce").fillna(-1)
    pc.sort_values(["country_code", "shop_id", "item_id", "date", "_total"], ascending=[True, True, True, True, False], inplace=True)
    pc_one = pc.groupby(PRODUCT_KEY, as_index=False).first()
    _assert_many_to_one(pc_one, PRODUCT_KEY, "aggregated product_categories")
    latest = latest.merge(
        pc_one[PRODUCT_KEY + ["category_id", "display_name"]].rename(
            columns={"category_id": "primary_shop_category_id", "display_name": "shop_category"}
        ),
        on=PRODUCT_KEY, how="left", validate="one_to_one",
    )
    product_category_join = _join_rate(latest, "primary_shop_category_id")

    audit = {
        "tables_loaded": {
            "products": [int(len(products_raw)), int(products_raw.shape[1])],
            "shop_info": [int(len(shops)), int(shops.shape[1])],
            "category_platform": [int(len(platform)), int(platform.shape[1])],
            "product_categories": [int(len(product_categories)), int(product_categories.shape[1])],
            "category_list": [int(len(shop_categories)), int(shop_categories.shape[1])],
        },
        "snapshot_dates": sorted(products["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "exact_duplicates_removed": duplicate_rows,
        "known_vn_shop_289646907_duplicates": known_duplicates,
        "deduplicated_snapshot_rows": int(len(products)),
        "latest_rows": int(len(latest)),
        "shop_join": shop_join,
        "platform_category_join": platform_join,
        "product_category_child_to_category_match_rate": product_category_child_match,
        "latest_product_category_join": product_category_join,
    }
    tables = {
        "products_raw": products_raw, "products_deduplicated": products, "shops": shops,
        "platform_categories": platform, "product_categories": product_categories,
        "shop_categories": shop_categories,
    }
    return latest, audit, tables
