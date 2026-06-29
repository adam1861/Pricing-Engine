from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from WebApp import service


def test_build_elasticity_table_preserves_meals_without_observations() -> None:
    train = pd.DataFrame(
        [
            {"meal_id": 1, "week": 1, "num_orders": 100, "checkout_price": 10.0},
            {"meal_id": 1, "week": 2, "num_orders": 50, "checkout_price": 20.0},
            {"meal_id": 2, "week": 1, "num_orders": 40, "checkout_price": 12.0},
        ]
    )

    elasticity = service._build_elasticity_table(train)

    meal_one = elasticity.loc[elasticity["meal_id"] == 1].iloc[0]
    meal_two = elasticity.loc[elasticity["meal_id"] == 2].iloc[0]

    assert set(elasticity["meal_id"]) == {1, 2}
    assert meal_one["observations"] == 1
    assert np.isclose(meal_one["median_elasticity"], -0.5)
    assert meal_two["observations"] == 0
    assert pd.isna(meal_two["median_elasticity"])


def test_load_data_rebuilds_elasticity_from_raw_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "Data" / "raw"
    processed_dir = tmp_path / "Data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "week": 1,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 10.0,
                "base_price": 12.0,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "num_orders": 100,
            },
            {
                "week": 2,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 12.0,
                "base_price": 12.0,
                "emailer_for_promotion": 1,
                "homepage_featured": 0,
                "num_orders": 80,
            },
            {
                "week": 1,
                "center_id": 20,
                "meal_id": 2,
                "checkout_price": 9.0,
                "base_price": 11.0,
                "emailer_for_promotion": 0,
                "homepage_featured": 1,
                "num_orders": 60,
            },
        ]
    ).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame(
        [
            {"meal_id": 1, "category": "Main", "cuisine": "Italian"},
            {"meal_id": 2, "category": "Snack", "cuisine": "Indian"},
        ]
    ).to_csv(raw_dir / "meal_info.csv", index=False)
    pd.DataFrame(
        [
            {"center_id": 10, "center_type": "TYPE_A"},
            {"center_id": 20, "center_type": "TYPE_B"},
        ]
    ).to_csv(raw_dir / "fulfilment_center_info.csv", index=False)

    monkeypatch.setattr(service, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(service, "TRAIN_PATH", raw_dir / "train.csv")
    monkeypatch.setattr(service, "MEAL_INFO_PATH", raw_dir / "meal_info.csv")
    monkeypatch.setattr(service, "CENTER_INFO_PATH", raw_dir / "fulfilment_center_info.csv")
    monkeypatch.setattr(service, "ELASTICITY_PATH", processed_dir / "avg_elasticity_per_meal.csv")
    service._load_data_cached.cache_clear()

    data, elasticity, meals, centers = service.load_data()

    assert "revenue" in data.columns
    assert len(data) == 3
    assert len(meals) == 2
    assert len(centers) == 2
    assert set(elasticity["meal_id"]) == {1, 2}
    assert (processed_dir / "avg_elasticity_per_meal.csv").exists() is False


def test_load_data_downloads_raw_files_from_env_urls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    raw_dir = tmp_path / "Data" / "raw"
    processed_dir = tmp_path / "Data" / "processed"

    pd.DataFrame(
        [
            {
                "week": 1,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 10.0,
                "base_price": 12.0,
                "emailer_for_promotion": 0,
                "homepage_featured": 0,
                "num_orders": 100,
            },
            {
                "week": 2,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 12.0,
                "base_price": 12.0,
                "emailer_for_promotion": 1,
                "homepage_featured": 0,
                "num_orders": 80,
            },
        ]
    ).to_csv(source_dir / "train.csv", index=False)
    pd.DataFrame(
        [{"meal_id": 1, "category": "Main", "cuisine": "Italian"}]
    ).to_csv(source_dir / "meal_info.csv", index=False)
    pd.DataFrame(
        [{"center_id": 10, "center_type": "TYPE_A"}]
    ).to_csv(source_dir / "fulfilment_center_info.csv", index=False)

    monkeypatch.setattr(service, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(service, "TRAIN_PATH", raw_dir / "train.csv")
    monkeypatch.setattr(service, "MEAL_INFO_PATH", raw_dir / "meal_info.csv")
    monkeypatch.setattr(service, "CENTER_INFO_PATH", raw_dir / "fulfilment_center_info.csv")
    monkeypatch.setattr(service, "ELASTICITY_PATH", processed_dir / "avg_elasticity_per_meal.csv")
    monkeypatch.setenv("DATA_BASE_URL", source_dir.as_uri() + "/")
    service._load_data_cached.cache_clear()

    data, elasticity, meals, centers = service.load_data()

    assert len(data) == 2
    assert len(elasticity) == 1
    assert len(meals) == 1
    assert len(centers) == 1
    assert raw_dir.joinpath("train.csv").exists()
    assert raw_dir.joinpath("meal_info.csv").exists()
    assert raw_dir.joinpath("fulfilment_center_info.csv").exists()


def test_build_recommendations_and_price_curve(monkeypatch) -> None:
    data = pd.DataFrame(
        [
            {
                "week": 1,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 10.0,
                "num_orders": 100,
                "revenue": 1000.0,
                "category": "Main",
                "cuisine": "Italian",
                "center_type": "TYPE_A",
            },
            {
                "week": 2,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 11.0,
                "num_orders": 95,
                "revenue": 1045.0,
                "category": "Main",
                "cuisine": "Italian",
                "center_type": "TYPE_A",
            },
            {
                "week": 3,
                "center_id": 10,
                "meal_id": 1,
                "checkout_price": 12.0,
                "num_orders": 90,
                "revenue": 1080.0,
                "category": "Main",
                "cuisine": "Italian",
                "center_type": "TYPE_A",
            },
            {
                "week": 1,
                "center_id": 20,
                "meal_id": 2,
                "checkout_price": 8.0,
                "num_orders": 60,
                "revenue": 480.0,
                "category": "Snack",
                "cuisine": "Indian",
                "center_type": "TYPE_B",
            },
            {
                "week": 2,
                "center_id": 20,
                "meal_id": 2,
                "checkout_price": 8.5,
                "num_orders": 58,
                "revenue": 493.0,
                "category": "Snack",
                "cuisine": "Indian",
                "center_type": "TYPE_B",
            },
            {
                "week": 3,
                "center_id": 20,
                "meal_id": 2,
                "checkout_price": 9.0,
                "num_orders": 55,
                "revenue": 495.0,
                "category": "Snack",
                "cuisine": "Indian",
                "center_type": "TYPE_B",
            },
        ]
    )
    elasticity = pd.DataFrame(
        [
            {
                "meal_id": 1,
                "avg_elasticity": -1.0,
                "median_elasticity": -1.0,
                "observations": 2,
            },
            {
                "meal_id": 2,
                "avg_elasticity": np.nan,
                "median_elasticity": np.nan,
                "observations": 0,
            },
        ]
    )
    meals = pd.DataFrame(
        [
            {"meal_id": 1, "category": "Main", "cuisine": "Italian"},
            {"meal_id": 2, "category": "Snack", "cuisine": "Indian"},
        ]
    )
    centers = pd.DataFrame(
        [
            {"center_id": 10, "center_type": "TYPE_A"},
            {"center_id": 20, "center_type": "TYPE_B"},
        ]
    )

    monkeypatch.setattr(service, "load_data", lambda: (data, elasticity, meals, centers))

    recommendations = service.build_recommendations(
        recent_weeks=2,
        max_change=0.10,
        candidate_count=5,
    )

    assert len(recommendations) == 2
    assert recommendations["revenue_uplift"].is_monotonic_decreasing
    assert set(recommendations["candidate_count"]) == {5}

    fallback_row = recommendations.loc[recommendations["meal_id"] == 2].iloc[0]
    assert fallback_row["median_elasticity"] == -1.0

    curve = service.build_price_curve(recommendations, meal_id=1)
    assert len(curve) == 5
    assert set(curve.columns) == {
        "candidate_price",
        "predicted_orders",
        "expected_revenue",
        "meal_id",
    }


def test_filter_helpers_reduce_rows_without_copying_extra_columns() -> None:
    data = pd.DataFrame(
        [
            {"week": 1, "category": "Main", "cuisine": "Italian"},
            {"week": 2, "category": "Main", "cuisine": "Indian"},
            {"week": 3, "category": "Snack", "cuisine": "Indian"},
        ]
    )
    recommendations = pd.DataFrame(
        [
            {"meal_id": 1, "category": "Main", "cuisine": "Italian"},
            {"meal_id": 2, "category": "Snack", "cuisine": "Indian"},
        ]
    )

    filtered_data = service.filter_data(
        data,
        category="Main",
        cuisine="Indian",
        week_start=2,
        week_end=2,
    )
    filtered_recommendations = service.filter_recommendations(
        recommendations,
        category="Snack",
        cuisine="Indian",
    )

    assert filtered_data.to_dict(orient="records") == [
        {"week": 2, "category": "Main", "cuisine": "Indian"}
    ]
    assert filtered_recommendations.to_dict(orient="records") == [
        {"meal_id": 2, "category": "Snack", "cuisine": "Indian"}
    ]
