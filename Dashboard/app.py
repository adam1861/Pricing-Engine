from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Pricing Engine Studio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "Data" / "raw"
PROCESSED_DIR = ROOT_DIR / "Data" / "processed"

MEAL_INFO_PATH = RAW_DIR / "meal_info.csv"
CENTER_INFO_PATH = RAW_DIR / "fulfilment_center_info.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "data.csv"
ELASTICITY_PATH = PROCESSED_DIR / "avg_elasticity_per_meal.csv"

ACCENT = "#B8522B"
INK = "#16202A"
MINT = "#A7C4BC"
SLATE = "#55626E"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(184, 82, 43, 0.10), transparent 34%),
                    radial-gradient(circle at top right, rgba(167, 196, 188, 0.18), transparent 28%),
                    linear-gradient(180deg, #fbf7f1 0%, #f6efe7 100%);
                color: {INK};
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }}
            .hero {{
                padding: 1.2rem 1.4rem;
                border-radius: 20px;
                background: linear-gradient(120deg, rgba(22, 32, 42, 0.95), rgba(68, 83, 95, 0.92));
                color: #f9f4ef;
                box-shadow: 0 18px 48px rgba(22, 32, 42, 0.16);
                margin-bottom: 1rem;
            }}
            .hero h1 {{
                margin: 0;
                font-size: 2.2rem;
                letter-spacing: 0.02em;
            }}
            .hero p {{
                margin: 0.5rem 0 0;
                font-size: 1rem;
                color: #d9e2e8;
                max-width: 52rem;
            }}
            .metric-card {{
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(22, 32, 42, 0.08);
                box-shadow: 0 12px 32px rgba(22, 32, 42, 0.08);
                min-height: 130px;
            }}
            .metric-label {{
                font-size: 0.88rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: {SLATE};
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: {INK};
                margin-top: 0.35rem;
            }}
            .metric-note {{
                margin-top: 0.45rem;
                color: {SLATE};
                font-size: 0.9rem;
            }}
            .section-note {{
                padding: 0.85rem 1rem;
                border-radius: 14px;
                background: rgba(167, 196, 188, 0.16);
                border-left: 4px solid {ACCENT};
                color: {INK};
                margin-bottom: 1rem;
            }}
            div[data-testid="stDataFrame"] {{
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(22, 32, 42, 0.08);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [
        MEAL_INFO_PATH,
        CENTER_INFO_PATH,
        PROCESSED_DATA_PATH,
        ELASTICITY_PATH,
    ]
    missing = [str(path.relative_to(ROOT_DIR)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required project files: " + ", ".join(missing)
        )

    processed = pd.read_csv(PROCESSED_DATA_PATH)
    elasticity = pd.read_csv(ELASTICITY_PATH)
    meals = pd.read_csv(MEAL_INFO_PATH)
    centers = pd.read_csv(CENTER_INFO_PATH)

    enriched = (
        processed.merge(meals, on="meal_id", how="left")
        .merge(centers[["center_id", "center_type"]], on="center_id", how="left")
    )

    return enriched, elasticity, meals, centers


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def build_recommendations(
    data: pd.DataFrame,
    elasticity: pd.DataFrame,
    meals: pd.DataFrame,
    recent_weeks: int,
    max_change: float,
    candidate_count: int,
) -> pd.DataFrame:
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

    recommendations = pd.DataFrame(records).sort_values("revenue_uplift", ascending=False)
    return recommendations


def build_price_curve(
    recommendations: pd.DataFrame,
    meal_id: int,
    candidate_count: int,
) -> pd.DataFrame:
    meal_row = recommendations.loc[recommendations["meal_id"] == meal_id].iloc[0]
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
    curve["meal_id"] = meal_id
    return curve


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Pricing Engine Studio</h1>
            <p>
                Interactive view of the food pricing dataset, meal-level elasticity estimates,
                and price recommendations derived from the current notebook workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data, elasticity, meals, centers = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    week_min = int(data["week"].min())
    week_max = int(data["week"].max())
    categories = sorted(meals["category"].dropna().unique().tolist())
    cuisines = sorted(meals["cuisine"].dropna().unique().tolist())

    st.sidebar.header("Controls")
    recent_weeks = st.sidebar.slider("Recent weeks for baseline", min_value=2, max_value=12, value=4)
    max_change_pct = st.sidebar.slider("Max price change (%)", min_value=5, max_value=30, value=20, step=1)
    candidate_count = st.sidebar.slider("Candidate prices per meal", min_value=10, max_value=50, value=20, step=5)
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)
    selected_cuisines = st.sidebar.multiselect("Cuisines", cuisines, default=cuisines)
    selected_week_range = st.sidebar.slider(
        "Explorer week range",
        min_value=week_min,
        max_value=week_max,
        value=(week_min, week_max),
    )

    recommendations = build_recommendations(
        data=data,
        elasticity=elasticity,
        meals=meals,
        recent_weeks=recent_weeks,
        max_change=max_change_pct / 100,
        candidate_count=candidate_count,
    )

    filtered_recommendations = recommendations[
        recommendations["category"].isin(selected_categories)
        & recommendations["cuisine"].isin(selected_cuisines)
    ].copy()

    filtered_data = data[
        data["category"].isin(selected_categories)
        & data["cuisine"].isin(selected_cuisines)
        & data["week"].between(selected_week_range[0], selected_week_range[1])
    ].copy()

    st.markdown(
        f"""
        <div class="section-note">
            Current grain: one row represents one meal in one center for one week.
            The recommendation logic follows the notebooks by aggregating demand and average price
            at the <strong>meal_id x week</strong> level before optimizing revenue.
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, recommendation_tab, explorer_tab = st.tabs(
        ["Overview", "Recommendation Lab", "Dataset Explorer"]
    )

    with overview_tab:
        total_revenue = filtered_data["revenue"].sum()
        total_orders = filtered_data["num_orders"].sum()
        total_rows = len(filtered_data)
        total_meals = filtered_data["meal_id"].nunique()

        card_cols = st.columns(4)
        with card_cols[0]:
            render_metric_card("Rows in scope", format_int(total_rows), "Filtered by sidebar category, cuisine, and week range.")
        with card_cols[1]:
            render_metric_card("Meals in scope", format_int(total_meals), "Unique products represented in the current filtered slice.")
        with card_cols[2]:
            render_metric_card("Orders in scope", format_int(total_orders), "Observed demand across all selected weeks and centers.")
        with card_cols[3]:
            render_metric_card("Revenue in scope", format_currency(total_revenue), "Historical checkout price multiplied by observed orders.")

        weekly_summary = (
            filtered_data.groupby("week", as_index=False)
            .agg(total_orders=("num_orders", "sum"), total_revenue=("revenue", "sum"))
            .sort_values("week")
        )

        category_summary = (
            filtered_data.groupby("category", as_index=False)
            .agg(total_orders=("num_orders", "sum"), total_revenue=("revenue", "sum"))
            .sort_values("total_revenue", ascending=False)
        )

        center_summary = (
            filtered_data.groupby("center_type", as_index=False)
            .agg(total_orders=("num_orders", "sum"), total_revenue=("revenue", "sum"))
            .sort_values("total_orders", ascending=False)
        )

        col_left, col_right = st.columns((1.3, 1))
        with col_left:
            weekly_fig = px.line(
                weekly_summary,
                x="week",
                y="total_orders",
                markers=True,
                title="Weekly demand trend",
                color_discrete_sequence=[ACCENT],
            )
            weekly_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.65)",
                margin=dict(l=16, r=16, t=48, b=16),
            )
            st.plotly_chart(weekly_fig, use_container_width=True)

        with col_right:
            center_fig = px.bar(
                center_summary,
                x="center_type",
                y="total_orders",
                title="Orders by center type",
                color="center_type",
                color_discrete_sequence=[ACCENT, MINT, "#D8B08C"],
            )
            center_fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.65)",
                margin=dict(l=16, r=16, t=48, b=16),
            )
            st.plotly_chart(center_fig, use_container_width=True)

        category_fig = px.bar(
            category_summary,
            x="total_revenue",
            y="category",
            orientation="h",
            title="Revenue by category",
            color="total_revenue",
            color_continuous_scale=["#E8D4BF", ACCENT],
        )
        category_fig.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            margin=dict(l=16, r=16, t=48, b=16),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(category_fig, use_container_width=True)

    with recommendation_tab:
        if filtered_recommendations.empty:
            st.warning("No recommendations match the current category/cuisine filters.")
        else:
            uplift_sum = filtered_recommendations["revenue_uplift"].sum()
            current_revenue_sum = filtered_recommendations["current_revenue"].sum()
            projected_revenue_sum = filtered_recommendations[
                "expected_revenue_at_recommended_price"
            ].sum()
            positive_share = (
                (filtered_recommendations["revenue_uplift"] > 0).mean() * 100
            )

            rec_cards = st.columns(4)
            with rec_cards[0]:
                render_metric_card(
                    "Meals optimized",
                    format_int(len(filtered_recommendations)),
                    "Meal-level price recommendations after category and cuisine filtering.",
                )
            with rec_cards[1]:
                render_metric_card(
                    "Current baseline revenue",
                    format_currency(current_revenue_sum),
                    f"Computed from the last {recent_weeks} weeks used for the baseline.",
                )
            with rec_cards[2]:
                render_metric_card(
                    "Projected revenue",
                    format_currency(projected_revenue_sum),
                    "Best expected revenue across the tested price grid for each meal.",
                )
            with rec_cards[3]:
                render_metric_card(
                    "Revenue uplift",
                    format_currency(uplift_sum),
                    f"{positive_share:.1f}% of meals show a positive uplift under the current settings.",
                )

            st.download_button(
                label="Download recommendations as CSV",
                data=filtered_recommendations.to_csv(index=False).encode("utf-8"),
                file_name="price_recommendations_streamlit.csv",
                mime="text/csv",
            )

            styled_recommendations = filtered_recommendations[
                [
                    "meal_id",
                    "category",
                    "cuisine",
                    "current_price",
                    "recommended_price",
                    "median_elasticity",
                    "current_orders",
                    "predicted_orders_at_recommended_price",
                    "current_revenue",
                    "expected_revenue_at_recommended_price",
                    "revenue_uplift",
                    "revenue_uplift_pct",
                    "observations",
                ]
            ].copy()

            st.dataframe(
                styled_recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "current_price": st.column_config.NumberColumn(format="%.2f"),
                    "recommended_price": st.column_config.NumberColumn(format="%.2f"),
                    "median_elasticity": st.column_config.NumberColumn(format="%.3f"),
                    "current_orders": st.column_config.NumberColumn(format="%.0f"),
                    "predicted_orders_at_recommended_price": st.column_config.NumberColumn(format="%.0f"),
                    "current_revenue": st.column_config.NumberColumn(format="%.2f"),
                    "expected_revenue_at_recommended_price": st.column_config.NumberColumn(format="%.2f"),
                    "revenue_uplift": st.column_config.NumberColumn(format="%.2f"),
                    "revenue_uplift_pct": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

            meal_options = filtered_recommendations.sort_values(
                ["revenue_uplift", "meal_id"], ascending=[False, True]
            )[["meal_id", "category", "cuisine"]]
            meal_labels = {
                row.meal_id: f"{row.meal_id} | {row.category} | {row.cuisine}"
                for row in meal_options.itertuples(index=False)
            }
            selected_meal = st.selectbox(
                "Inspect one recommendation curve",
                options=meal_options["meal_id"].tolist(),
                format_func=lambda meal_id: meal_labels[meal_id],
            )

            meal_row = filtered_recommendations.loc[
                filtered_recommendations["meal_id"] == selected_meal
            ].iloc[0]
            curve = build_price_curve(filtered_recommendations, selected_meal, candidate_count)

            curve_fig = go.Figure()
            curve_fig.add_trace(
                go.Scatter(
                    x=curve["candidate_price"],
                    y=curve["expected_revenue"],
                    mode="lines+markers",
                    name="Expected revenue",
                    line=dict(color=ACCENT, width=3),
                )
            )
            curve_fig.add_trace(
                go.Scatter(
                    x=curve["candidate_price"],
                    y=curve["predicted_orders"],
                    mode="lines+markers",
                    name="Predicted orders",
                    yaxis="y2",
                    line=dict(color=MINT, width=3),
                )
            )
            curve_fig.add_vline(
                x=meal_row["recommended_price"],
                line_dash="dash",
                line_color=INK,
            )
            curve_fig.update_layout(
                title=f"Price-response curve for meal {selected_meal}",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.65)",
                margin=dict(l=16, r=16, t=48, b=16),
                yaxis=dict(title="Expected revenue"),
                yaxis2=dict(title="Predicted orders", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )

            detail_cols = st.columns((1.4, 1))
            with detail_cols[0]:
                st.plotly_chart(curve_fig, use_container_width=True)
            with detail_cols[1]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Meal snapshot</div>
                        <div class="metric-note">Meal ID: <strong>{int(meal_row["meal_id"])}</strong></div>
                        <div class="metric-note">Category: <strong>{meal_row["category"]}</strong></div>
                        <div class="metric-note">Cuisine: <strong>{meal_row["cuisine"]}</strong></div>
                        <div class="metric-note">Current price: <strong>{format_currency(meal_row["current_price"])}</strong></div>
                        <div class="metric-note">Recommended price: <strong>{format_currency(meal_row["recommended_price"])}</strong></div>
                        <div class="metric-note">Median elasticity: <strong>{meal_row["median_elasticity"]:.3f}</strong></div>
                        <div class="metric-note">Projected uplift: <strong>{format_currency(meal_row["revenue_uplift"])}</strong></div>
                        <div class="metric-note">Allowed range: <strong>{format_currency(meal_row["min_allowed_price"])} to {format_currency(meal_row["max_allowed_price"])}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with explorer_tab:
        st.dataframe(
            filtered_data[
                [
                    "week",
                    "center_id",
                    "center_type",
                    "meal_id",
                    "category",
                    "cuisine",
                    "checkout_price",
                    "base_price",
                    "num_orders",
                    "revenue",
                    "emailer_for_promotion",
                    "homepage_featured",
                ]
            ].sort_values(["week", "center_id", "meal_id"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "checkout_price": st.column_config.NumberColumn(format="%.2f"),
                "base_price": st.column_config.NumberColumn(format="%.2f"),
                "revenue": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.caption(
            f"Dataset snapshot covers {format_int(filtered_data['center_id'].nunique())} centers, "
            f"{format_int(filtered_data['meal_id'].nunique())} meals, and weeks "
            f"{selected_week_range[0]} to {selected_week_range[1]}."
        )


if __name__ == "__main__":
    main()
