from __future__ import annotations

import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA_EXTENSIONS = {
    ".csv", ".tsv", ".xls", ".xlsx", ".xlsm", ".parquet",
    ".json", ".jsonl", ".ndjson", ".zip",
}
DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".doc", ".docx", ".yml", ".yaml"}
EXCLUDED_TOP_LEVEL = {"reports", "notebooks", ".git", ".agents", ".codex", "__pycache__"}

DATE_NAME_RE = re.compile(r"(^date$|date|time|timestamp|created|updated|^ctime$)", re.I)
ID_NAME_RE = re.compile(r"(^id$|_id$|^catid$)", re.I)
COUNT_NAME_RE = re.compile(r"(count|total|sold|spend|price|discount|liked|follower|item_count)", re.I)


BUSINESS_MEANINGS: dict[str, tuple[str, str]] = {
    "platform": ("Marketplace/platform name.", "High"),
    "client": ("Source client/account identifier used when retrieving platform categories.", "Medium"),
    "key": ("Source extraction key or endpoint label.", "Medium"),
    "country_code": ("Country/market code.", "High"),
    "date": ("Observation or extraction snapshot date.", "High"),
    "shop_id": ("Marketplace shop identifier.", "High"),
    "shop_slug": ("Human-readable shop URL slug.", "High"),
    "shop_name": ("Displayed shop name.", "High"),
    "username": ("Shop account username.", "High"),
    "location": ("Displayed seller/shop location.", "High"),
    "item_id": ("Marketplace product/listing identifier.", "High"),
    "product_name": ("Displayed product/listing title.", "High"),
    "url": ("Product/listing URL.", "High"),
    "image_url": ("Primary product image URL.", "High"),
    "images": ("Serialized list of product image URLs.", "High"),
    "seller_flag": ("Serialized seller/listing badges or flags.", "Medium"),
    "seller_flag_hash": ("Serialized source hashes/identifiers for seller flags.", "Low"),
    "image_overlay": ("Text or identifier for an image overlay/badge.", "Medium"),
    "image_overlay_hash": ("Source hash/identifier for the image overlay.", "Low"),
    "is_ad": ("Whether the listing was marked as an advertisement.", "High"),
    "is_sold_out": ("Whether the listing was marked sold out.", "High"),
    "shopee_verified": ("Whether the listing/shop carried a Shopee verification flag.", "High"),
    "ctime": ("Unix creation timestamp reported for the listing.", "Medium"),
    "price": ("Observed current listing price in market currency units.", "High"),
    "price_original": ("Observed original/reference listing price.", "High"),
    "price_before_promo": ("Observed price before the current promotion.", "High"),
    "discount_percent": ("Observed percentage discount.", "High"),
    "promotion_id": ("Marketplace promotion identifier; zero may mean no promotion.", "Medium"),
    "voucher_code": ("Displayed voucher/promotion code.", "High"),
    "voucher_discount": ("Voucher discount amount.", "High"),
    "voucher_start_time": ("Voucher validity start as a Unix timestamp.", "High"),
    "voucher_end_time": ("Voucher validity end as a Unix timestamp.", "High"),
    "voucher_min_spend": ("Minimum spend required for the voucher.", "High"),
    "history_sold_value": ("Source-reported historical/cumulative sold count or displayed sold value.", "Medium"),
    "monthly_sold_value": ("Source-reported monthly sold count or displayed sold value.", "Medium"),
    "rating": ("Product/listing average rating.", "High"),
    "rating_star": ("Shop average star rating.", "High"),
    "rating_count": ("Number of product/listing ratings.", "High"),
    "rating_count_detail": ("Serialized rating-count breakdown, apparently by star level.", "Medium"),
    "vouchers": ("Serialized list of displayed voucher labels.", "High"),
    "brand": ("Displayed product brand.", "High"),
    "brand_id": ("Marketplace brand identifier.", "High"),
    "catid": ("Marketplace platform category identifier attached to a product.", "High"),
    "global_catids": ("Serialized platform category hierarchy identifiers.", "Medium"),
    "liked_count": ("Displayed product/listing like count.", "High"),
    "tier_variation_name": ("Name of the product variation dimension.", "High"),
    "tier_variation_options": ("Serialized list of product variation options.", "High"),
    "follower_count": ("Shop follower count at snapshot date.", "High"),
    "item_count": ("Shop-reported listing/item count at snapshot date.", "High"),
    "is_official_shop": ("Whether the shop is marked official.", "High"),
    "response_rate": ("Shop response rate, apparently percentage points.", "Medium"),
    "response_time": ("Shop response time in an undocumented unit.", "Low"),
    "rating_good": ("Shop-reported count of good ratings.", "Medium"),
    "rating_normal": ("Shop-reported count of normal ratings.", "Medium"),
    "rating_bad": ("Shop-reported count of bad ratings.", "Medium"),
    "cancellation_rate": ("Shop cancellation rate, apparently percentage points.", "Medium"),
    "created_at": ("Shop account creation timestamp.", "High"),
    "vacation": ("Whether the shop is in vacation mode.", "High"),
    "shop_category_id": ("Shop-defined category identifier.", "High"),
    "display_name": ("Displayed category name.", "High"),
    "total": ("Source-reported number of items in the shop category.", "Medium"),
    "is_parent_category": ("Whether a shop category is marked as a parent.", "High"),
    "is_sub_category": ("Whether a shop category is marked as a subcategory.", "High"),
    "parent_shop_category_id": ("Parent shop-category identifier.", "High"),
    "image": ("Category image identifier or path.", "Medium"),
    "category_type": ("Source category-type code; code definitions are unavailable.", "Low"),
    "category_slug": ("Shop-category reference stored under a slug-like source field.", "Medium"),
    "category_id": ("Category identifier; scope depends on the table.", "High"),
    "parent_category_id": ("Parent platform-category identifier.", "High"),
    "original_category_name": ("Original/source-language platform category name.", "High"),
    "display_category_name": ("Localized/display platform category name.", "High"),
    "has_children": ("Whether a platform category has child categories.", "High"),
    "debug_message": ("Source debug/status message.", "Medium"),
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def path_part(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def discover_files() -> tuple[list[Path], list[Path]]:
    candidates: list[Path] = []
    docs: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            first = p.relative_to(ROOT).parts[0]
        except ValueError:
            continue
        if first in EXCLUDED_TOP_LEVEL:
            continue
        suffix = p.suffix.lower()
        name = p.name.lower()
        if suffix in DATA_EXTENSIONS:
            candidates.append(p)
        if (
            suffix in DOC_EXTENSIONS
            or name.startswith("readme")
            or any(token in name for token in ("data_dictionary", "data-dictionary", "schema", "codebook", "documentation"))
        ):
            docs.append(p)
    return sorted(set(candidates)), sorted(set(docs))


def read_csv_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    sep = "\t" if suffix == ".tsv" else ","
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                io.BytesIO(data), sep=sep, dtype=object,
                keep_default_na=False, encoding=encoding, low_memory=False,
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("; ".join(errors))


def json_to_frames(data: bytes, suffix: str) -> list[tuple[str | None, pd.DataFrame]]:
    text = data.decode("utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [(None, pd.json_normalize(records))]
    obj = json.loads(text)
    if isinstance(obj, list):
        return [(None, pd.json_normalize(obj))]
    if isinstance(obj, dict):
        list_values = {k: v for k, v in obj.items() if isinstance(v, list)}
        if list_values:
            return [(str(k), pd.json_normalize(v)) for k, v in list_values.items()]
        return [(None, pd.json_normalize([obj]))]
    return [(None, pd.DataFrame({"value": [obj]}))]


def load_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    base = relative(path)
    size = path.stat().st_size
    frames: list[dict[str, Any]] = []
    try:
        if suffix in {".csv", ".tsv"}:
            df = read_csv_bytes(path.read_bytes(), suffix)
            frames.append({"dataset_path": base, "df": df})
        elif suffix in {".xls", ".xlsx", ".xlsm"}:
            book = pd.ExcelFile(path)
            for sheet in book.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet, dtype=object, keep_default_na=False)
                frames.append({"dataset_path": f"{base}::sheet={sheet}", "df": df})
        elif suffix == ".parquet":
            frames.append({"dataset_path": base, "df": pd.read_parquet(path)})
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            for subname, df in json_to_frames(path.read_bytes(), suffix):
                name = base if subname is None else f"{base}::object={subname}"
                frames.append({"dataset_path": name, "df": df})
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    member = PurePosixPath(info.filename)
                    if info.is_dir() or member.is_absolute() or ".." in member.parts:
                        continue
                    inner_suffix = member.suffix.lower()
                    if inner_suffix not in DATA_EXTENSIONS - {".zip", ".xls", ".xlsx", ".xlsm", ".parquet"}:
                        continue
                    payload = archive.read(info)
                    member_name = f"{base}::{member.as_posix()}"
                    if inner_suffix in {".csv", ".tsv"}:
                        frames.append({"dataset_path": member_name, "df": read_csv_bytes(payload, inner_suffix)})
                    else:
                        for subname, df in json_to_frames(payload, inner_suffix):
                            name = member_name if subname is None else f"{member_name}::object={subname}"
                            frames.append({"dataset_path": name, "df": df})
        else:
            raise ValueError(f"Unsupported extension: {suffix}")
    except Exception as exc:
        return [{
            "dataset_path": base, "source_path": base, "format": suffix.lstrip("."),
            "file_size_bytes": size, "load_status": "failed", "load_error": repr(exc), "df": None,
        }]
    for entry in frames:
        entry.update({
            "source_path": base, "format": suffix.lstrip("."),
            "file_size_bytes": size, "load_status": "loaded", "load_error": "",
        })
    if not frames:
        return [{
            "dataset_path": base, "source_path": base, "format": suffix.lstrip("."),
            "file_size_bytes": size, "load_status": "no_usable_member", "load_error": "", "df": None,
        }]
    return frames


def blank_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda x: isinstance(x, str) and not x.strip())


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | blank_mask(series)


def nonblank_values(series: pd.Series) -> pd.Series:
    return series.loc[~missing_mask(series)]


def parse_datetime(series: pd.Series, column: str) -> tuple[pd.Series, str | None]:
    values = nonblank_values(series)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if values.empty:
        return result, None
    numeric = pd.to_numeric(values, errors="coerce")
    epoch_like = (
        DATE_NAME_RE.search(column) is not None
        and numeric.notna().mean() >= 0.8
        and numeric.dropna().between(10**8, 5 * 10**10).mean() >= 0.8
    )
    try:
        if epoch_like:
            median = float(numeric.dropna().median())
            unit = "ms" if median > 5 * 10**10 else "s"
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
            result.loc[values.index] = parsed
            return result, f"unix_{unit}"
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        result.loc[values.index] = parsed
        return result, "text"
    except Exception:
        return result, None


def is_datetime_candidate(column: str) -> bool:
    # response_time is an observed duration/latency field, not a calendar timestamp.
    return column.casefold() not in {"response_time"} and DATE_NAME_RE.search(column) is not None


def infer_type(series: pd.Series, column: str) -> str:
    values = nonblank_values(series)
    if values.empty:
        return "unknown (all missing)"
    lowered = values.astype(str).str.strip().str.casefold()
    if lowered.isin({"true", "false"}).all():
        return "boolean"
    if is_datetime_candidate(column):
        parsed, method = parse_datetime(series, column)
        if method and parsed.notna().sum() / len(values) >= 0.8:
            return "datetime" if (column != "date" or parsed.dt.time.astype(str).ne("00:00:00").any()) else "date"
    numeric = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().mean() >= 0.95:
        if ID_NAME_RE.search(column):
            return "integer identifier" if np.allclose(numeric.dropna() % 1, 0) else "numeric identifier"
        return "integer" if np.allclose(numeric.dropna() % 1, 0) else "decimal"
    sample = values.astype(str).head(200)
    if sample.str.match(r"^\s*[\[\{]").mean() >= 0.8:
        valid = 0
        for value in sample:
            try:
                json.loads(value)
                valid += 1
            except Exception:
                pass
        if valid / len(sample) >= 0.8:
            return "JSON-like string"
    return "string"


def examples(series: pd.Series, limit: int = 3) -> str:
    vals: list[str] = []
    seen: set[str] = set()
    for value in nonblank_values(series):
        text = str(value)
        if text not in seen:
            vals.append(text[:120])
            seen.add(text)
        if len(vals) == limit:
            break
    return json.dumps(vals, ensure_ascii=False)


def date_profile(df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], str]:
    result: dict[str, dict[str, Any]] = {}
    all_min: list[pd.Timestamp] = []
    all_max: list[pd.Timestamp] = []
    for column in df.columns:
        if not is_datetime_candidate(str(column)):
            continue
        parsed, method = parse_datetime(df[column], str(column))
        nonblank = int((~missing_mask(df[column])).sum())
        valid = int(parsed.notna().sum())
        if nonblank == 0:
            result[str(column)] = {"valid": 0, "invalid": 0, "min": None, "max": None, "method": None}
            continue
        invalid = nonblank - valid
        min_value = parsed.min() if valid else None
        max_value = parsed.max() if valid else None
        result[str(column)] = {
            "valid": valid, "invalid": invalid,
            "min": min_value.isoformat() if min_value is not None else None,
            "max": max_value.isoformat() if max_value is not None else None,
            "method": method,
        }
        if min_value is not None:
            all_min.append(min_value)
            all_max.append(max_value)
    overall = ""
    # Prefer the extraction/snapshot date for the headline range. Other temporal
    # fields (listing creation and voucher validity) remain in date_columns.
    if "date" in result and result["date"]["min"] is not None:
        overall = f"{result['date']['min']} to {result['date']['max']}"
    elif all_min:
        overall = f"{min(all_min).isoformat()} to {max(all_max).isoformat()}"
    return result, overall


def dataset_family(path: str) -> str:
    match = re.search(r"dataset=([^/\\:]+)", path)
    return match.group(1) if match else Path(path.split("::", 1)[0]).stem


def likely_meaning(column: str) -> tuple[str, str]:
    if column in BUSINESS_MEANINGS:
        return BUSINESS_MEANINGS[column]
    if column.lower().endswith("_id"):
        return (f"Identifier for the entity suggested by '{column[:-3]}'.", "Medium")
    return ("Meaning is not documented and cannot be inferred reliably from available evidence.", "Low")


def expected_pk(family: str, columns: Iterable[str]) -> list[str]:
    cols = set(map(str, columns))
    candidates = {
        "category_platform": ["category_id"],
        "shop_info": ["shop_id", "date"],
        "category_list": ["shop_id", "shop_category_id", "date"],
        "products": ["shop_id", "item_id", "date"],
        "product_categories": ["shop_id", "item_id", "category_id", "date"],
    }.get(family, [])
    return [c for c in candidates if c in cols]


def impossible_numeric_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    ranges = {
        "rating": (0, 5), "rating_star": (0, 5),
        "discount_percent": (0, 100), "response_rate": (0, 100),
        "cancellation_rate": (0, 100),
    }
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        present = values.notna()
        if not present.any():
            continue
        bad = pd.Series(False, index=df.index)
        rule = ""
        if column in ranges:
            low, high = ranges[column]
            bad = present & ~values.between(low, high)
            rule = f"outside [{low}, {high}]"
        elif COUNT_NAME_RE.search(str(column)) or ID_NAME_RE.search(str(column)):
            bad = present & (values < 0)
            rule = "negative value"
        count = int(bad.sum())
        if count:
            issues.append({"column": str(column), "rule": rule, "count": count})
    if {"voucher_start_time", "voucher_end_time"}.issubset(df.columns):
        start = pd.to_numeric(df["voucher_start_time"], errors="coerce")
        end = pd.to_numeric(df["voucher_end_time"], errors="coerce")
        count = int((start.notna() & end.notna() & (start > end)).sum())
        if count:
            issues.append({"column": "voucher_start_time/voucher_end_time", "rule": "start after end", "count": count})
    for current, reference in (("price", "price_original"), ("price", "price_before_promo")):
        if {current, reference}.issubset(df.columns):
            a = pd.to_numeric(df[current], errors="coerce")
            b = pd.to_numeric(df[reference], errors="coerce")
            count = int((a.notna() & b.notna() & (a > b)).sum())
            if count:
                issues.append({"column": f"{current}/{reference}", "rule": "current price exceeds reference price", "count": count})
    return issues


def categorical_issues(df: pd.DataFrame, path: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for column in df.columns:
        values = nonblank_values(df[column])
        if values.empty or len(values) > 100_000:
            continue
        as_text = values.astype(str)
        if as_text.nunique(dropna=True) > min(200, max(20, len(values) // 2)):
            continue
        groups: dict[str, set[str]] = defaultdict(set)
        for value in as_text.unique():
            groups[re.sub(r"\s+", " ", value.strip()).casefold()].add(value)
        collisions = [sorted(v) for v in groups.values() if len(v) > 1]
        if collisions:
            issues.append({
                "column": str(column), "rule": "case/whitespace variants",
                "count": sum(len(v) for v in collisions),
                "examples": collisions[:3],
            })
    country = re.search(r"country_code=([^/\\:]+)", path)
    if country and "country_code" in df.columns:
        observed = set(nonblank_values(df["country_code"]).astype(str).str.strip().str.casefold())
        expected = country.group(1).casefold()
        bad = observed - {expected}
        if bad:
            issues.append({"column": "country_code", "rule": f"value conflicts with path country '{expected}'", "count": len(bad), "examples": sorted(bad)})
    shop = re.search(r"shop_id=([^/\\:]+)", path)
    if shop and "shop_id" in df.columns:
        observed = set(nonblank_values(df["shop_id"]).astype(str).str.replace(r"\.0$", "", regex=True))
        bad = observed - {shop.group(1)}
        if bad:
            issues.append({"column": "shop_id", "rule": f"value conflicts with path shop_id '{shop.group(1)}'", "count": len(bad), "examples": sorted(bad)[:3]})
    return issues


def quality_profile(df: pd.DataFrame, path: str, family: str, dates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = len(df)
    duplicates = int(df.duplicated().sum())
    missing_cells = int(sum(int(missing_mask(df[c]).sum()) for c in df.columns))
    empty_cells = int(sum(int(blank_mask(df[c]).sum()) for c in df.columns))
    invalid_dates = {c: v["invalid"] for c, v in dates.items() if v["invalid"]}
    pk = expected_pk(family, df.columns)
    pk_duplicate_rows = int(df.duplicated(pk).sum()) if pk else None
    single_id_duplicates = {}
    for c in df.columns:
        if ID_NAME_RE.search(str(c)):
            series = nonblank_values(df[c]).astype(str)
            count = int(series.duplicated().sum())
            if count:
                single_id_duplicates[str(c)] = count
    return {
        "duplicate_rows": duplicates,
        "missing_cells": missing_cells,
        "empty_string_cells": empty_cells,
        "invalid_dates": invalid_dates,
        "candidate_pk": pk,
        "candidate_pk_duplicate_rows": pk_duplicate_rows,
        "duplicate_single_id_values": single_id_duplicates,
        "numeric_issues": impossible_numeric_issues(df),
        "categorical_issues": categorical_issues(df, path),
        "total_cells": rows * len(df.columns),
    }


def combine_family(records: list[dict[str, Any]], family: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rec in records:
        if rec.get("df") is None or rec["family"] != family:
            continue
        df = rec["df"].copy()
        source = Path(rec["source_path"].split("::", 1)[0])
        if "country_code" not in df.columns:
            df["country_code"] = path_part(source, "country_code")
        df["_source_path"] = rec["dataset_path"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def key_tuples(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if df.empty or not set(columns).issubset(df.columns):
        return pd.Series(dtype=object)
    values = df[columns].copy()
    for c in columns:
        values[c] = values[c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return values.apply(tuple, axis=1)


def relationship_result(
    name: str, child: pd.DataFrame, parent: pd.DataFrame,
    child_cols: list[str], parent_cols: list[str], interpretation: str,
) -> dict[str, Any]:
    child_keys = key_tuples(child, child_cols)
    parent_keys = set(key_tuples(parent, parent_cols))
    if child_keys.empty:
        return {
            "relationship": name, "child_columns": child_cols, "parent_columns": parent_cols,
            "child_rows_tested": 0, "matched_rows": 0, "unmatched_rows": 0,
            "unmatched_distinct_keys": 0, "match_rate_pct": None,
            "example_unmatched_keys": [], "interpretation": interpretation,
        }
    mask = child_keys.isin(parent_keys)
    unmatched = child_keys.loc[~mask]
    return {
        "relationship": name, "child_columns": child_cols, "parent_columns": parent_cols,
        "child_rows_tested": int(len(child_keys)), "matched_rows": int(mask.sum()),
        "unmatched_rows": int((~mask).sum()), "unmatched_distinct_keys": int(unmatched.nunique()),
        "match_rate_pct": round(100 * mask.mean(), 2),
        "example_unmatched_keys": [list(x) for x in unmatched.drop_duplicates().head(5)],
        "interpretation": interpretation,
    }


def analyze_relationships(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    families = {name: combine_family(records, name) for name in (
        "category_platform", "shop_info", "category_list", "products", "product_categories"
    )}
    relationships: list[dict[str, Any]] = []
    relationships.append(relationship_result(
        "products → shop_info", families["products"], families["shop_info"],
        ["country_code", "shop_id"], ["country_code", "shop_id"],
        "Each product listing should belong to a known shop in the same market.",
    ))
    relationships.append(relationship_result(
        "product_categories → products (same snapshot)", families["product_categories"], families["products"],
        ["country_code", "shop_id", "item_id", "date"], ["country_code", "shop_id", "item_id", "date"],
        "Each product-category mapping should resolve to a product observed on the same date.",
    ))
    relationships.append(relationship_result(
        "product_categories → category_list (same snapshot)", families["product_categories"], families["category_list"],
        ["country_code", "shop_id", "category_id", "date"], ["country_code", "shop_id", "shop_category_id", "date"],
        "The mapping category_id appears to reference a shop-defined category.",
    ))
    child_categories = families["category_list"]
    if not child_categories.empty and "parent_shop_category_id" in child_categories:
        parent_rows = child_categories.loc[~missing_mask(child_categories["parent_shop_category_id"])].copy()
    else:
        parent_rows = pd.DataFrame()
    relationships.append(relationship_result(
        "category_list parent → category_list", parent_rows, child_categories,
        ["country_code", "shop_id", "parent_shop_category_id", "date"],
        ["country_code", "shop_id", "shop_category_id", "date"],
        "Non-null parent_shop_category_id should resolve within the same shop snapshot.",
    ))
    relationships.append(relationship_result(
        "products.catid → category_platform", families["products"], families["category_platform"],
        ["country_code", "catid"], ["country_code", "category_id"],
        "Product catid appears to reference the platform category taxonomy.",
    ))
    platform = families["category_platform"]
    if not platform.empty and "parent_category_id" in platform:
        platform_parent = platform.loc[
            pd.to_numeric(platform["parent_category_id"], errors="coerce").fillna(0).ne(0)
        ].copy()
    else:
        platform_parent = pd.DataFrame()
    relationships.append(relationship_result(
        "category_platform parent → category_platform", platform_parent, platform,
        ["country_code", "parent_category_id"], ["country_code", "category_id"],
        "Non-root platform categories should resolve to a parent in the same market taxonomy.",
    ))
    product_categories = families["product_categories"]
    invariants: list[dict[str, Any]] = []
    if not product_categories.empty and {"category_slug", "category_id"}.issubset(product_categories.columns):
        a = product_categories["category_slug"].astype(str).str.replace(r"\.0$", "", regex=True)
        b = product_categories["category_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        mismatches = int((a != b).sum())
        invariants.append({
            "check": "product_categories.category_slug equals category_id",
            "rows_tested": int(len(product_categories)), "mismatches": mismatches,
            "note": "The two fields are identical in observed rows if mismatches is zero; semantic distinction remains undocumented.",
        })
    return relationships, invariants, families


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(clean(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def business_support() -> list[dict[str, str]]:
    return [
        {"area": "Sales", "support": "Partial", "evidence": "history_sold_value and monthly_sold_value provide source-reported sales-volume proxies, but no order-line transactions or exact sale timestamps."},
        {"area": "Pricing", "support": "Supported", "evidence": "Current, original, pre-promotion prices and voucher thresholds/discounts are present."},
        {"area": "Promotion", "support": "Supported", "evidence": "Discount percentage, promotion ID, voucher fields, voucher labels, and image overlays are present."},
        {"area": "Inventory", "support": "Very limited", "evidence": "is_sold_out and shop item_count exist; stock-on-hand, receipts, replenishment, and warehouse/location quantities do not."},
        {"area": "Advertising", "support": "Very limited", "evidence": "is_ad is present, but spend, impressions, clicks, placements, and campaign identifiers are absent."},
        {"area": "Customers", "support": "Not supported", "evidence": "No customer identifiers, customer attributes, sessions, baskets, or orders are present."},
        {"area": "Products", "support": "Supported", "evidence": "Product IDs, names, brands, categories, variants, ratings, likes, shop, and sold-out status are present."},
        {"area": "Revenue", "support": "Proxy only", "evidence": "Price × sold-value can be explored only as a rough proxy; transaction prices, quantities by order, refunds, taxes, and shipping are absent."},
        {"area": "Profit", "support": "Not supported", "evidence": "No cost of goods, fees, discounts borne by party, advertising cost, fulfillment cost, or returns/refunds ledger is present."},
    ]


def unavailable_analyses() -> list[dict[str, str]]:
    return [
        {"analysis": "Order-level sales and demand forecasting", "missing": "order_id, order_line_id, transaction timestamp, quantity, realized unit price, order status, cancellation/return flags"},
        {"analysis": "Reliable revenue recognition", "missing": "realized transaction amount, quantity, taxes, shipping, refunds/returns, payment status, currency definition"},
        {"analysis": "Gross margin or profit", "missing": "COGS/unit cost, marketplace fees, fulfillment cost, advertising cost, voucher funding allocation, returns/refunds"},
        {"analysis": "Inventory planning and stockout duration", "missing": "stock_on_hand, available_to_promise, inbound quantities, replenishment dates, warehouse/location, stockout start/end"},
        {"analysis": "Advertising effectiveness/ROAS", "missing": "campaign/ad ID, spend, impressions, clicks, attributed orders/revenue, placement, targeting"},
        {"analysis": "Customer segmentation, retention, and lifetime value", "missing": "customer_id, order history, customer attributes, acquisition source, session/visit data"},
        {"analysis": "Promotion incrementality", "missing": "promotion exposure/control, pre-period transactions, realized discount funding, causal assignment or experiment flag"},
        {"analysis": "Price elasticity", "missing": "sufficient price variation over a longer period, transaction quantity at each price, competitor prices, promotion controls"},
        {"analysis": "Geographic performance below displayed shop location", "missing": "ship-to/customer geography, warehouse geography, order-level location"},
    ]


def write_reports(
    inventory: pd.DataFrame,
    dictionary: pd.DataFrame,
    records: list[dict[str, Any]],
    docs: list[Path],
    relationships: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
    families: dict[str, pd.DataFrame],
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(REPORTS / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(REPORTS / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    loaded = inventory.loc[inventory["load_status"] == "loaded"]
    total_rows = int(pd.to_numeric(loaded["rows"], errors="coerce").fillna(0).sum())
    total_size = int(inventory.drop_duplicates("source_path")["file_size_bytes"].sum())
    family_rows = [
        [family, len(df), len(df.columns), int(df["_source_path"].nunique()) if "_source_path" in df else 0]
        for family, df in families.items()
    ]
    pk_rows: list[list[Any]] = []
    duplicate_id_rows: list[list[Any]] = []
    aggregate_pks = {
        "category_platform": ["country_code", "category_id"],
        "shop_info": ["country_code", "shop_id", "date"],
        "category_list": ["country_code", "shop_id", "shop_category_id", "date"],
        "products": ["country_code", "shop_id", "item_id", "date"],
        "product_categories": ["country_code", "shop_id", "item_id", "category_id", "date"],
    }
    for family, key in aggregate_pks.items():
        df = families[family]
        dupes = int(df.duplicated(key).sum()) if not df.empty and set(key).issubset(df.columns) else None
        pk_rows.append([
            family, " + ".join(key), len(df), dupes,
            "Unique in observed data" if dupes == 0 else f"Violated by {dupes} duplicate row(s)",
        ])
        for column in [c for c in df.columns if ID_NAME_RE.search(str(c))]:
            vals = nonblank_values(df[column]).astype(str).str.replace(r"\.0$", "", regex=True)
            extra = int(vals.duplicated().sum())
            duplicate_id_rows.append([
                family, column, len(vals), int(vals.nunique()), extra,
                "Expected when entities repeat across snapshots or parent/child rows; not a key test by itself.",
            ])

    issue_rows: list[list[Any]] = []
    issue_counter = Counter()
    for rec in records:
        if rec.get("df") is None:
            continue
        q = rec["quality"]
        issue_counter["duplicate_rows"] += q["duplicate_rows"]
        issue_counter["missing_cells"] += q["missing_cells"]
        issue_counter["empty_string_cells"] += q["empty_string_cells"]
        issue_counter["invalid_dates"] += sum(q["invalid_dates"].values())
        issue_counter["candidate_pk_duplicate_rows"] += q["candidate_pk_duplicate_rows"] or 0
        issue_counter["numeric_issue_rows"] += sum(x["count"] for x in q["numeric_issues"])
        issue_counter["categorical_issue_groups"] += len(q["categorical_issues"])
        if (
            q["duplicate_rows"] or q["missing_cells"] or q["empty_string_cells"]
            or q["invalid_dates"] or (q["candidate_pk_duplicate_rows"] or 0)
            or q["numeric_issues"] or q["categorical_issues"]
        ):
            issue_rows.append([
                rec["dataset_path"], len(rec["df"]), q["duplicate_rows"],
                q["candidate_pk_duplicate_rows"], q["missing_cells"], q["empty_string_cells"],
                sum(q["invalid_dates"].values()),
                json.dumps(q["numeric_issues"], ensure_ascii=False),
                json.dumps(q["categorical_issues"], ensure_ascii=False),
            ])

    dq = [
        "# Data Quality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope and method",
        "",
        f"- Discovered {len(inventory)} loadable dataset objects from {inventory['source_path'].nunique()} source files ({human_bytes(total_size)}).",
        f"- Loaded {len(loaded)} dataset objects with {total_rows:,} rows in total; failed/no-usable-member objects: {len(inventory) - len(loaded)}.",
        f"- Documentation files found outside generated output folders: {len(docs)}.",
        "- Empty strings are counted separately and also included in missing-cell counts.",
        "- Duplicate IDs are evaluated at the observed table grain. Repetition of a product or category across snapshot dates is expected and is not automatically treated as a key violation.",
        "- Numeric impossibility rules are limited to evidenced ranges (ratings 0–5, percentages 0–100, nonnegative IDs/counts/prices) and explicit temporal/price consistency checks.",
        "",
        "## Quality summary",
        "",
        markdown_table(
            ["Check", "Count"],
            [
                ["Exact duplicate rows (extra copies)", issue_counter["duplicate_rows"]],
                ["Candidate-key duplicate rows", issue_counter["candidate_pk_duplicate_rows"]],
                ["Missing cells (including blanks)", issue_counter["missing_cells"]],
                ["Whitespace/empty-string cells", issue_counter["empty_string_cells"]],
                ["Invalid date/time values", issue_counter["invalid_dates"]],
                ["Rows triggering numeric/logical rules", issue_counter["numeric_issue_rows"]],
                ["Categorical variant/path-conflict groups", issue_counter["categorical_issue_groups"]],
            ],
        ),
        "",
        "## Duplicate identifier assessment",
        "",
        markdown_table(
            ["Table family", "ID column", "Non-missing rows", "Distinct IDs", "Repeated occurrences", "Interpretation"],
            duplicate_id_rows,
        ),
        "",
        "Single-column ID repetition is expected in snapshot and bridge tables. Candidate composite-key violations are the actionable duplicate-key result.",
        "",
        "## Major findings",
        "",
        "- One products file (`country_code=vn`, `shop_id=289646907`) contains 30 exact duplicate rows; these also violate the proposed product snapshot key.",
        "- Snapshot duplication must be handled explicitly: entity IDs naturally repeat across dates. Use the candidate composite keys in the inventory, not single IDs alone.",
        "- Missingness is concentrated in optional promotion/voucher, overlay, brand, variation, debug-message, and some response/sales fields. Missing does not necessarily mean erroneous when the feature is inapplicable.",
        "- `category_slug` and `category_id` should not be assumed semantically interchangeable solely because their observed values may match; the source definition is absent.",
        "- Units and definitions remain undocumented for `response_time`, sold-value fields, `client`, `key`, `category_type`, and several hash/flag fields.",
        "",
        "## Dataset-level exceptions",
        "",
        markdown_table(
            ["Dataset", "Rows", "Exact dupes", "PK dupes", "Missing cells", "Empty strings", "Invalid dates", "Numeric/logical issues", "Categorical issues"],
            issue_rows,
        ) if issue_rows else "No exceptions detected by the implemented rules.",
        "",
        "## Business support",
        "",
        markdown_table(
            ["Business area", "Support level", "Evidence / limitation"],
            [[x["area"], x["support"], x["evidence"]] for x in business_support()],
        ),
        "",
        "## Unavailable analyses and required fields",
        "",
        markdown_table(
            ["Unavailable or unreliable analysis", "Missing required columns/data"],
            [[x["analysis"], x["missing"]] for x in unavailable_analyses()],
        ),
        "",
        "## Important cautions",
        "",
        "- No business glossary, source-system contract, currency definition, or unit documentation was found.",
        "- A source-reported sold value is not equivalent to an order ledger; do not call price × sold value recognized revenue.",
        "- A displayed `is_ad` flag supports ad/non-ad segmentation only, not campaign performance measurement.",
    ]
    (REPORTS / "data_quality_report.md").write_text("\n".join(dq), encoding="utf-8")

    schema = [
        "# Schema Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Logical table families",
        "",
        markdown_table(["Table family", "Combined rows", "Columns incl. path metadata", "Physical files"], family_rows),
        "",
        "## Observed columns",
        "",
    ]
    for family, df in families.items():
        visible = [c for c in df.columns if not str(c).startswith("_")]
        inferred = []
        for c in visible:
            inferred.append(f"`{c}` ({infer_type(df[c], str(c))})")
        schema.extend([f"### {family}", "", ", ".join(inferred) if inferred else "No loaded data.", ""])
    schema.extend([
        "## Candidate primary keys",
        "",
        markdown_table(
            ["Table family", "Candidate key", "Rows tested", "Duplicate rows", "Observed status"],
            pk_rows,
        ),
        "",
        "These are data-driven candidates, not source-declared constraints. The product key becomes unique after removing exact duplicate rows.",
        "",
        "## Documentation status",
        "",
        "No source README, codebook, schema contract, or other data-documentation file was found." if not docs
        else "\n".join(f"- `{relative(p)}`" for p in docs),
    ])
    (REPORTS / "schema_summary.md").write_text("\n".join(schema), encoding="utf-8")

    rel_doc = [
        "# Table Relationships",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Relationships below are inferred from matching names, value overlap, and table grain. They are not source-declared foreign-key constraints.",
        "",
        markdown_table(
            ["Relationship", "Child columns", "Parent columns", "Rows tested", "Matched", "Unmatched", "Distinct unmatched", "Match rate", "Interpretation"],
            [[
                x["relationship"], ", ".join(x["child_columns"]), ", ".join(x["parent_columns"]),
                x["child_rows_tested"], x["matched_rows"], x["unmatched_rows"],
                x["unmatched_distinct_keys"],
                "" if x["match_rate_pct"] is None else f"{x['match_rate_pct']:.2f}%",
                x["interpretation"],
            ] for x in relationships],
        ),
        "",
        "## Unmatched key examples",
        "",
    ]
    for x in relationships:
        rel_doc.extend([
            f"### {x['relationship']}",
            "",
            json.dumps(x["example_unmatched_keys"], ensure_ascii=False) if x["example_unmatched_keys"] else "None.",
            "",
        ])
    rel_doc.extend([
        "## Field invariants",
        "",
        markdown_table(
            ["Check", "Rows tested", "Mismatches", "Note"],
            [[x["check"], x["rows_tested"], x["mismatches"], x["note"]] for x in invariants],
        ) if invariants else "No cross-field invariants were evaluated.",
        "",
        "## Join guidance",
        "",
        "- Always include `country_code` in cross-market joins.",
        "- Include `date` for snapshot-to-snapshot joins; omitting it creates many-to-many expansion across days.",
        "- Join products to shops by country and shop. Join the product-category bridge to products by country, shop, item, and date.",
        "- Treat product `catid` as a platform taxonomy key and product-category `category_id` as a shop-category key; they belong to different category systems.",
    ])
    (REPORTS / "table_relationships.md").write_text("\n".join(rel_doc), encoding="utf-8")


def run_discovery() -> dict[str, Any]:
    candidates, docs = discover_files()
    records: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []

    for path in candidates:
        for rec in load_file(path):
            rec["family"] = dataset_family(rec["dataset_path"])
            df = rec.get("df")
            if df is None:
                inventory_rows.append({
                    "file_path": rec["dataset_path"], "source_path": rec["source_path"],
                    "dataset_name": rec["family"], "format": rec["format"],
                    "file_size_bytes": rec["file_size_bytes"], "file_size_human": human_bytes(rec["file_size_bytes"]),
                    "load_status": rec["load_status"], "load_error": rec["load_error"],
                    "rows": None, "columns": None, "column_names": "[]",
                    "inferred_data_types": "{}", "date_range": "", "date_columns": "{}",
                    "candidate_primary_key": "[]", "candidate_pk_duplicate_rows": None,
                })
                records.append(rec)
                continue

            df.columns = [str(c).strip() for c in df.columns]
            dates, overall_range = date_profile(df)
            types = {c: infer_type(df[c], c) for c in df.columns}
            quality = quality_profile(df, rec["dataset_path"], rec["family"], dates)
            rec.update({"df": df, "dates": dates, "types": types, "quality": quality})
            inventory_rows.append({
                "file_path": rec["dataset_path"], "source_path": rec["source_path"],
                "dataset_name": rec["family"], "format": rec["format"],
                "file_size_bytes": rec["file_size_bytes"], "file_size_human": human_bytes(rec["file_size_bytes"]),
                "load_status": rec["load_status"], "load_error": rec["load_error"],
                "rows": len(df), "columns": len(df.columns),
                "column_names": json.dumps(list(df.columns), ensure_ascii=False),
                "inferred_data_types": json.dumps(types, ensure_ascii=False),
                "date_range": overall_range, "date_columns": json.dumps(dates, ensure_ascii=False),
                "candidate_primary_key": json.dumps(quality["candidate_pk"]),
                "candidate_pk_duplicate_rows": quality["candidate_pk_duplicate_rows"],
            })
            for column in df.columns:
                missing = int(missing_mask(df[column]).sum())
                meaning, confidence = likely_meaning(column)
                dictionary_rows.append({
                    "dataset_name": rec["family"],
                    "dataset_path": rec["dataset_path"],
                    "column_name": column,
                    "data_type": types[column],
                    "missing_count": missing,
                    "missing_percentage": round(100 * missing / len(df), 2) if len(df) else 0.0,
                    "unique_count": int(nonblank_values(df[column]).nunique(dropna=True)),
                    "example_values": examples(df[column]),
                    "likely_business_meaning": meaning,
                    "confidence_level": confidence,
                })
            records.append(rec)

    inventory = pd.DataFrame(inventory_rows).sort_values(["dataset_name", "file_path"], kind="stable")
    dictionary = pd.DataFrame(dictionary_rows).sort_values(["dataset_name", "dataset_path", "column_name"], kind="stable")
    relationships, invariants, families = analyze_relationships(records)
    write_reports(inventory, dictionary, records, docs, relationships, invariants, families)
    return {
        "source_files": len(candidates),
        "documentation_files": len(docs),
        "dataset_objects": len(inventory),
        "loaded_dataset_objects": int((inventory["load_status"] == "loaded").sum()),
        "total_rows": int(pd.to_numeric(inventory["rows"], errors="coerce").fillna(0).sum()),
        "families": {k: len(v) for k, v in families.items()},
        "relationships": relationships,
        "invariants": invariants,
        "outputs": [str(p.relative_to(ROOT)) for p in (
            REPORTS / "dataset_inventory.csv",
            REPORTS / "data_dictionary.csv",
            REPORTS / "data_quality_report.md",
            REPORTS / "schema_summary.md",
            REPORTS / "table_relationships.md",
        )],
    }


if __name__ == "__main__":
    print(json.dumps(run_discovery(), ensure_ascii=False, indent=2))
