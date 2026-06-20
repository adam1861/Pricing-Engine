from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_existing_dir(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = _resolve_existing_dir([ROOT_DIR / "Data", ROOT_DIR / "data"])
RAW_DIR = _resolve_existing_dir([DATA_DIR / "raw", DATA_DIR / "Raw"])
PROCESSED_DIR = _resolve_existing_dir([DATA_DIR / "processed", DATA_DIR / "Processed"])

MEAL_INFO_PATH = RAW_DIR / "meal_info.csv"
CENTER_INFO_PATH = RAW_DIR / "fulfilment_center_info.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "data.csv"
ELASTICITY_PATH = PROCESSED_DIR / "avg_elasticity_per_meal.csv"


@lru_cache(maxsize=1)
def _load_data_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [
        MEAL_INFO_PATH,
        CENTER_INFO_PATH,
        PROCESSED_DATA_PATH,
        ELASTICITY_PATH,
    ]
    missing = [str(path.relative_to(ROOT_DIR)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required project files: " + ", ".join(missing))

    processed = pd.read_csv(PROCESSED_DATA_PATH)
    elasticity = pd.read_csv(ELASTICITY_PATH)
    meals = pd.read_csv(MEAL_INFO_PATH)
    centers = pd.read_csv(CENTER_INFO_PATH)

    enriched = (
        processed.merge(meals, on="meal_id", how="left")
        .merge(centers[["center_id", "center_type"]], on="center_id", how="left")
    )

    return enriched, elasticity, meals, centers


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, elasticity, meals, centers = _load_data_cached()
    return data.copy(), elasticity.copy(), meals.copy(), centers.copy()


def get_filter_metadata() -> dict[str, object]:
    data, _, meals, _ = load_data()
    return {
        "week_min": int(data["week"].min()),
        "week_max": int(data["week"].max()),
        "categories": sorted(meals["category"].dropna().unique().tolist()),
        "cuisines": sorted(meals["cuisine"].dropna().unique().tolist()),
    }


def build_recommendations(
    recent_weeks: int = 4,
    max_change: float = 0.20,
    candidate_count: int = 20,
) -> pd.DataFrame:
    data, elasticity, meals, _ = load_data()

    last_week = int(data["week"].max())
    recent_cutoff = max(int(data["week"].min()), last_week - recent_weeks + 1)
    recent_data = data[data["week"] >= recent_cutoff].copy()

    recent_weekly_meals = (
        recent_data.groupby(["meal_id", "week"], as_index=False)
        .agg(
            weekly_orders=("num_orders", "sum"),
            weekly_price=("checkout_price", "mean"),
        )
        .sort_values(["meal_id", "week"])
    )

    baseline = (
        recent_weekly_meals.groupby("meal_id", as_index=False)
        .agg(
            current_price=("weekly_price", "median"),
            current_orders=("weekly_orders", "mean"),
        )
    )

    price_bounds = data.groupby("meal_id", as_index=False).agg(
        historical_min_price=("checkout_price", lambda x: x.quantile(0.05)),
        historical_max_price=("checkout_price", lambda x: x.quantile(0.95)),
    )

    elasticity_table = elasticity[["meal_id", "median_elasticity", "observations"]].copy()
    recommendation_data = (
        baseline.merge(price_bounds, on="meal_id", how="left")
        .merge(elasticity_table, on="meal_id", how="left")
        .merge(meals, on="meal_id", how="left")
    )

    fallback_elasticity = elasticity_table["median_elasticity"].dropna().median()
    if pd.isna(fallback_elasticity):
        fallback_elasticity = -1.0

    recommendation_data["median_elasticity"] = recommendation_data["median_elasticity"].fillna(
        fallback_elasticity
    )
    recommendation_data["observations"] = recommendation_data["observations"].fillna(0).astype(int)
    recommendation_data["change_min_price"] = recommendation_data["current_price"] * (1 - max_change)
    recommendation_data["change_max_price"] = recommendation_data["current_price"] * (1 + max_change)
    recommendation_data["min_allowed_price"] = recommendation_data[
        ["historical_min_price", "change_min_price"]
    ].max(axis=1)
    recommendation_data["max_allowed_price"] = recommendation_data[
        ["historical_max_price", "change_max_price"]
    ].min(axis=1)
    recommendation_data["range_valid"] = (
        recommendation_data["min_allowed_price"] < recommendation_data["max_allowed_price"]
    )

    records: list[dict[str, float | int | str]] = []
    for row in recommendation_data.itertuples(index=False):
        if (
            not row.range_valid
            or pd.isna(row.current_price)
            or row.current_price <= 0
            or pd.isna(row.current_orders)
        ):
            candidate_prices = np.array([row.current_price], dtype=float)
        else:
            candidate_prices = np.linspace(
                row.min_allowed_price,
                row.max_allowed_price,
                candidate_count,
            )

        predicted_orders = row.current_orders * (
            candidate_prices / row.current_price
        ) ** row.median_elasticity
        predicted_orders = np.clip(predicted_orders, a_min=0, a_max=None)
        expected_revenues = candidate_prices * predicted_orders

        best_index = int(np.argmax(expected_revenues))
        current_revenue = row.current_price * row.current_orders
        recommended_price = float(candidate_prices[best_index])
        predicted_orders_at_recommended_price = float(predicted_orders[best_index])
        expected_revenue_at_recommended_price = float(expected_revenues[best_index])
        revenue_uplift = expected_revenue_at_recommended_price - current_revenue
        revenue_uplift_pct = (
            (revenue_uplift / current_revenue) * 100 if current_revenue else 0.0
        )

        records.append(
            {
                "meal_id": int(row.meal_id),
                "category": row.category,
                "cuisine": row.cuisine,
                "current_price": float(row.current_price),
                "recommended_price": recommended_price,
                "current_orders": float(row.current_orders),
                "predicted_orders_at_recommended_price": predicted_orders_at_recommended_price,
                "current_revenue": float(current_revenue),
                "expected_revenue_at_recommended_price": expected_revenue_at_recommended_price,
                "revenue_uplift": float(revenue_uplift),
                "revenue_uplift_pct": float(revenue_uplift_pct),
                "historical_min_price": float(row.historical_min_price),
                "historical_max_price": float(row.historical_max_price),
                "min_allowed_price": float(row.min_allowed_price),
                "max_allowed_price": float(row.max_allowed_price),
                "median_elasticity": float(row.median_elasticity),
                "observations": int(row.observations),
                "recent_weeks_used": int(recent_weeks),
                "recent_cutoff_week": int(recent_cutoff),
                "candidate_count": int(candidate_count),
            }
        )

    return pd.DataFrame(records).sort_values("revenue_uplift", ascending=False)


def filter_data(
    data: pd.DataFrame,
    category: str | None = None,
    cuisine: str | None = None,
    week_start: int | None = None,
    week_end: int | None = None,
) -> pd.DataFrame:
    filtered = data.copy()
    if category:
        filtered = filtered[filtered["category"] == category]
    if cuisine:
        filtered = filtered[filtered["cuisine"] == cuisine]
    if week_start is not None:
        filtered = filtered[filtered["week"] >= week_start]
    if week_end is not None:
        filtered = filtered[filtered["week"] <= week_end]
    return filtered


def filter_recommendations(
    recommendations: pd.DataFrame,
    category: str | None = None,
    cuisine: str | None = None,
) -> pd.DataFrame:
    filtered = recommendations.copy()
    if category:
        filtered = filtered[filtered["category"] == category]
    if cuisine:
        filtered = filtered[filtered["cuisine"] == cuisine]
    return filtered


def build_price_curve(recommendations: pd.DataFrame, meal_id: int) -> pd.DataFrame:
    meal_row = recommendations.loc[recommendations["meal_id"] == meal_id].iloc[0]
    candidate_count = int(meal_row["candidate_count"])
    candidate_prices = np.linspace(
        meal_row["min_allowed_price"],
        meal_row["max_allowed_price"],
        candidate_count,
    )
    predicted_orders = meal_row["current_orders"] * (
        candidate_prices / meal_row["current_price"]
    ) ** meal_row["median_elasticity"]
    predicted_orders = np.clip(predicted_orders, a_min=0, a_max=None)
    expected_revenues = candidate_prices * predicted_orders

    curve = pd.DataFrame(
        {
            "candidate_price": candidate_prices,
            "predicted_orders": predicted_orders,
            "expected_revenue": expected_revenues,
        }
    )
    curve["meal_id"] = int(meal_id)
    return curve
