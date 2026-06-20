from __future__ import annotations

from io import StringIO
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from WebApp.service import (
    build_price_curve,
    build_recommendations,
    filter_data,
    filter_recommendations,
    get_filter_metadata,
    load_data,
)


APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

ACCENT = "#B8522B"
INK = "#16202A"
MINT = "#A7C4BC"
SAND = "#E7D7C4"

app = FastAPI(title="Pricing Engine Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def format_compact(value: float, decimals: int = 1) -> str:
    abs_value = abs(value)
    suffixes = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for threshold, suffix in suffixes:
        if abs_value >= threshold:
            compact = value / threshold
            return f"{compact:.{decimals}f}{suffix}"
    if float(value).is_integer():
        return format_int(value)
    return f"{value:.{decimals}f}"


def build_empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=INK),
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=18, r=18, t=52, b=18),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def fig_to_html(fig: go.Figure) -> str:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=18, r=18, t=52, b=18),
        font=dict(color=INK),
    )
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


def build_overview_figures(filtered_data):
    if filtered_data.empty:
        empty = build_empty_figure("No data in current scope", "Adjust the filters to bring rows back into scope.")
        return fig_to_html(empty), fig_to_html(empty), fig_to_html(empty)

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

    weekly_fig = px.line(
        weekly_summary,
        x="week",
        y="total_orders",
        markers=True,
        title="Weekly demand trend",
        color_discrete_sequence=[ACCENT],
    )
    weekly_fig.update_traces(line=dict(width=3), marker=dict(size=7))
    weekly_fig.update_traces(
        hovertemplate="Week %{x}<br>Orders %{y:,.0f}<extra></extra>"
    )
    weekly_fig.update_yaxes(tickformat="~s")
    weekly_fig.update_xaxes(dtick=10)

    category_fig = px.bar(
        category_summary,
        x="total_revenue",
        y="category",
        orientation="h",
        title="Revenue by category",
        color="total_revenue",
        color_continuous_scale=[SAND, ACCENT],
    )
    category_fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
    category_fig.update_traces(
        hovertemplate="Category %{y}<br>Revenue %{x:,.2f}<extra></extra>"
    )
    category_fig.update_xaxes(tickformat="~s")

    center_fig = px.bar(
        center_summary,
        x="center_type",
        y="total_orders",
        title="Orders by center type",
        color="center_type",
        color_discrete_sequence=[ACCENT, MINT, "#D9B898"],
    )
    center_fig.update_layout(showlegend=False)
    center_fig.update_traces(
        hovertemplate="Center %{x}<br>Orders %{y:,.0f}<extra></extra>"
    )
    center_fig.update_yaxes(tickformat="~s")

    return fig_to_html(weekly_fig), fig_to_html(category_fig), fig_to_html(center_fig)


def build_curve_figure(filtered_recommendations, selected_meal: int | None) -> tuple[str, dict[str, str] | None]:
    if filtered_recommendations.empty or selected_meal is None:
        empty = build_empty_figure("Recommendation curve", "No recommendation is available for the current filter set.")
        return fig_to_html(empty), None

    meal_row = filtered_recommendations.loc[
        filtered_recommendations["meal_id"] == selected_meal
    ].iloc[0]
    curve = build_price_curve(filtered_recommendations, selected_meal)

    curve_fig = go.Figure()
    curve_fig.add_trace(
        go.Scatter(
            x=curve["candidate_price"],
            y=curve["expected_revenue"],
            mode="lines+markers",
            name="Expected revenue",
            line=dict(color=ACCENT, width=3),
            hovertemplate="Price %{x:,.2f}<br>Expected revenue %{y:,.2f}<extra></extra>",
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
            hovertemplate="Price %{x:,.2f}<br>Predicted orders %{y:,.0f}<extra></extra>",
        )
    )
    curve_fig.add_vline(x=meal_row["recommended_price"], line_dash="dash", line_color=INK)
    curve_fig.update_layout(
        title=f"Price-response curve for meal {selected_meal}",
        xaxis=dict(title="Candidate price", tickformat=",.2f"),
        yaxis=dict(title="Expected revenue", tickformat="~s"),
        yaxis2=dict(title="Predicted orders", overlaying="y", side="right", tickformat="~s"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    detail = {
        "meal_id": str(int(meal_row["meal_id"])),
        "category": str(meal_row["category"]),
        "cuisine": str(meal_row["cuisine"]),
        "current_price": format_currency(meal_row["current_price"]),
        "recommended_price": format_currency(meal_row["recommended_price"]),
        "median_elasticity": f'{meal_row["median_elasticity"]:.3f}',
        "revenue_uplift": format_currency(meal_row["revenue_uplift"]),
        "price_range": (
            f'{format_currency(meal_row["min_allowed_price"])} to '
            f'{format_currency(meal_row["max_allowed_price"])}'
        ),
    }
    return fig_to_html(curve_fig), detail


def serialize_recommendation_table(filtered_recommendations):
    rows = []
    for row in filtered_recommendations.head(20).itertuples(index=False):
        rows.append(
            {
                "meal_id": int(row.meal_id),
                "category": row.category,
                "cuisine": row.cuisine,
                "current_price": format_currency(row.current_price),
                "recommended_price": format_currency(row.recommended_price),
                "elasticity": f"{row.median_elasticity:.3f}",
                "current_revenue": format_currency(row.current_revenue),
                "projected_revenue": format_currency(row.expected_revenue_at_recommended_price),
                "uplift": format_currency(row.revenue_uplift),
                "uplift_pct": format_pct(row.revenue_uplift_pct),
                "uplift_class": "positive" if row.revenue_uplift >= 0 else "negative",
            }
        )
    return rows


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    category: str | None = Query(default=None),
    cuisine: str | None = Query(default=None),
    week_start: int | None = Query(default=None),
    week_end: int | None = Query(default=None),
    recent_weeks: int = Query(default=4, ge=2, le=12),
    max_change_pct: int = Query(default=20, ge=5, le=30),
    candidate_count: int = Query(default=20, ge=10, le=50),
    meal_id: int | None = Query(default=None),
):
    try:
        metadata = get_filter_metadata()
    except FileNotFoundError as exc:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "page_title": "Pricing Engine Studio",
                "error_title": "Project data files are missing on the server",
                "error_message": str(exc),
                "error_hint": (
                    "Render deployed the application code, but the CSV files expected in "
                    "Data/raw and Data/processed were not included in the deployed repo."
                ),
            },
            status_code=500,
        )
    week_min = metadata["week_min"]
    week_max = metadata["week_max"]

    if week_start is None:
        week_start = week_min
    if week_end is None:
        week_end = week_max

    week_start = clamp(week_start, week_min, week_max)
    week_end = clamp(week_end, week_min, week_max)
    if week_start > week_end:
        week_start, week_end = week_end, week_start

    data, _, _, _ = load_data()
    recommendations = build_recommendations(
        recent_weeks=recent_weeks,
        max_change=max_change_pct / 100,
        candidate_count=candidate_count,
    )

    filtered_data = filter_data(
        data,
        category=category,
        cuisine=cuisine,
        week_start=week_start,
        week_end=week_end,
    )
    filtered_recommendations = filter_recommendations(
        recommendations,
        category=category,
        cuisine=cuisine,
    )

    if not filtered_recommendations.empty:
        valid_meals = filtered_recommendations["meal_id"].tolist()
        if meal_id not in valid_meals:
            meal_id = int(filtered_recommendations.iloc[0]["meal_id"])
    else:
        meal_id = None

    weekly_plot_html, category_plot_html, center_plot_html = build_overview_figures(filtered_data)
    curve_plot_html, selected_meal_detail = build_curve_figure(filtered_recommendations, meal_id)

    total_revenue = filtered_data["revenue"].sum() if not filtered_data.empty else 0.0
    total_orders = filtered_data["num_orders"].sum() if not filtered_data.empty else 0.0
    total_rows = len(filtered_data)
    total_meals = filtered_data["meal_id"].nunique() if not filtered_data.empty else 0

    uplift_sum = filtered_recommendations["revenue_uplift"].sum() if not filtered_recommendations.empty else 0.0
    current_revenue_sum = (
        filtered_recommendations["current_revenue"].sum() if not filtered_recommendations.empty else 0.0
    )
    projected_revenue_sum = (
        filtered_recommendations["expected_revenue_at_recommended_price"].sum()
        if not filtered_recommendations.empty
        else 0.0
    )
    positive_share = (
        (filtered_recommendations["revenue_uplift"] > 0).mean() * 100
        if not filtered_recommendations.empty
        else 0.0
    )

    recommendation_meals = []
    if not filtered_recommendations.empty:
        for row in filtered_recommendations[["meal_id", "category", "cuisine"]].itertuples(index=False):
            recommendation_meals.append(
                {
                    "meal_id": int(row.meal_id),
                    "label": f"{int(row.meal_id)} | {row.category} | {row.cuisine}",
                    "selected": int(row.meal_id) == meal_id,
                }
            )

    raw_rows = []
    for row in filtered_data.head(18).itertuples(index=False):
        raw_rows.append(
            {
                "week": int(row.week),
                "center_id": int(row.center_id),
                "center_type": str(row.center_type),
                "meal_id": int(row.meal_id),
                "category": str(row.category),
                "cuisine": str(row.cuisine),
                "checkout_price": format_currency(row.checkout_price),
                "num_orders": format_int(row.num_orders),
                "revenue": format_currency(row.revenue),
            }
        )

    context = {
        "request": request,
        "page_title": "Pricing Engine Studio",
        "filters": {
            "category": category or "",
            "cuisine": cuisine or "",
            "week_start": week_start,
            "week_end": week_end,
            "recent_weeks": recent_weeks,
            "max_change_pct": max_change_pct,
            "candidate_count": candidate_count,
            "meal_id": meal_id,
        },
        "metadata": metadata,
        "overview_cards": [
            {
                "label": "Rows in scope",
                "value": format_compact(total_rows, decimals=0),
                "detail": format_int(total_rows),
                "note": "Filtered by current category, cuisine, and week controls.",
            },
            {
                "label": "Meals in scope",
                "value": format_compact(total_meals, decimals=0),
                "detail": format_int(total_meals),
                "note": "Unique products represented in the filtered dataset slice.",
            },
            {
                "label": "Orders in scope",
                "value": format_compact(total_orders),
                "detail": format_int(total_orders),
                "note": "Observed demand across the selected weeks and centers.",
            },
            {
                "label": "Revenue in scope",
                "value": format_compact(total_revenue),
                "detail": format_currency(total_revenue),
                "note": "Historical checkout price multiplied by observed orders.",
            },
        ],
        "recommendation_cards": [
            {
                "label": "Meals optimized",
                "value": format_compact(len(filtered_recommendations), decimals=0),
                "detail": format_int(len(filtered_recommendations)),
                "note": "Meal-level recommendations after category and cuisine filters.",
            },
            {
                "label": "Current baseline revenue",
                "value": format_compact(current_revenue_sum),
                "detail": format_currency(current_revenue_sum),
                "note": f"Baseline built from the last {recent_weeks} weeks.",
            },
            {
                "label": "Projected revenue",
                "value": format_compact(projected_revenue_sum),
                "detail": format_currency(projected_revenue_sum),
                "note": "Best expected revenue across the tested price grid.",
            },
            {
                "label": "Revenue uplift",
                "value": format_compact(uplift_sum),
                "detail": format_currency(uplift_sum),
                "note": f"{positive_share:.1f}% of meals show positive uplift.",
            },
        ],
        "weekly_plot_html": weekly_plot_html,
        "category_plot_html": category_plot_html,
        "center_plot_html": center_plot_html,
        "curve_plot_html": curve_plot_html,
        "selected_meal_detail": selected_meal_detail,
        "recommendation_meals": recommendation_meals,
        "recommendation_rows": serialize_recommendation_table(filtered_recommendations),
        "raw_rows": raw_rows,
        "raw_row_count": format_int(total_rows),
        "filtered_center_count": format_int(filtered_data["center_id"].nunique() if not filtered_data.empty else 0),
        "filtered_meal_count": format_int(filtered_data["meal_id"].nunique() if not filtered_data.empty else 0),
        "download_query": (
            f"category={category or ''}&cuisine={cuisine or ''}&recent_weeks={recent_weeks}"
            f"&max_change_pct={max_change_pct}&candidate_count={candidate_count}"
        ),
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/api/recommendations", response_class=JSONResponse)
def recommendations_api(
    category: str | None = Query(default=None),
    cuisine: str | None = Query(default=None),
    recent_weeks: int = Query(default=4, ge=2, le=12),
    max_change_pct: int = Query(default=20, ge=5, le=30),
    candidate_count: int = Query(default=20, ge=10, le=50),
):
    try:
        recommendations = build_recommendations(
            recent_weeks=recent_weeks,
            max_change=max_change_pct / 100,
            candidate_count=candidate_count,
        )
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    filtered_recommendations = filter_recommendations(
        recommendations,
        category=category,
        cuisine=cuisine,
    )
    return JSONResponse(filtered_recommendations.round(4).to_dict(orient="records"))


@app.get("/recommendations.csv")
def recommendations_csv(
    category: str | None = Query(default=None),
    cuisine: str | None = Query(default=None),
    recent_weeks: int = Query(default=4, ge=2, le=12),
    max_change_pct: int = Query(default=20, ge=5, le=30),
    candidate_count: int = Query(default=20, ge=10, le=50),
):
    try:
        recommendations = build_recommendations(
            recent_weeks=recent_weeks,
            max_change=max_change_pct / 100,
            candidate_count=candidate_count,
        )
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    filtered_recommendations = filter_recommendations(
        recommendations,
        category=category,
        cuisine=cuisine,
    )
    buffer = StringIO()
    filtered_recommendations.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=price_recommendations.csv"},
    )


@app.get("/health", response_class=JSONResponse)
def healthcheck():
    return {"status": "ok"}
