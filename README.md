# Pricing Engine

Notebook-driven pricing analysis for a weekly food demand dataset. The project prepares a merged dataset, explores demand patterns, estimates meal-level price elasticity, and recommends revenue-oriented prices through an interactive FastAPI interface.

## What Is In This Repo

- `Data/raw/`
  Source CSV files:
  - `train.csv`
  - `meal_info.csv`
  - `fulfilment_center_info.csv`
- `Data/processed/`
  Generated analysis files:
  - `data.csv`
  - `avg_elasticity_per_meal.csv`
- `Notebooks/`
  Analysis notebooks used in the project
- `WebApp/main.py`
  FastAPI web interface for exploring the dataset and price recommendations
- `api/index.py`
  Vercel-compatible FastAPI entrypoint that re-exports the main app
- `WebApp/templates/` and `WebApp/static/`
  HTML templates and styling for the FastAPI app
- `Dashboard/app.py`
  Legacy Streamlit dashboard kept as a fallback

## Dataset Grain

The main table is `Data/raw/train.csv`.

One row represents:

- one `meal_id`
- in one `center_id`
- for one `week`

So the same meal appears many times across weeks and centers.

Current dataset summary:

- `456,548` observations
- `51` unique meals
- `77` fulfillment centers
- `145` weeks

## Notebook Status

- `01_data_preparation.ipynb`
  Merges raw files, one-hot encodes categories, and writes `Data/processed/data.csv`
- `02_eda.ipynb`
  Basic exploratory data analysis on the processed dataset
- `03_feature_engineering.ipynb`
  Placeholder heading only
- `04_demand_forecasting.ipynb`
  Placeholder heading only
- `05_price_elasticity.ipynb`
  Aggregates data by `meal_id x week` and writes `Data/processed/avg_elasticity_per_meal.csv`
- `06_price_optimization.ipynb`
  Builds meal-level revenue recommendations from recent price/order behavior and elasticity

## Method Used Today

The current project logic is based on the notebooks already in the repo:

1. Merge order history with meal and fulfillment-center metadata.
2. Build a processed dataset with one-hot encoded categorical fields and a `revenue` column.
3. Aggregate by `meal_id` and `week` to estimate meal-level elasticity from percentage changes in price and orders.
4. Use recent weekly demand as a baseline.
5. Test candidate prices within historical and change-based bounds.
6. Select the price that maximizes expected revenue for each meal.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Web Interface

```bash
uvicorn WebApp.main:app --reload
```

The app includes:

- dataset overview and KPIs
- category, cuisine, and week filters
- meal-level price recommendations
- price-response curve inspection for one meal at a time
- downloadable recommendation table
- JSON recommendations endpoint at `/api/recommendations`

Open `http://127.0.0.1:8000`.

## Optional Legacy Dashboard

If you still want the older exploratory version:

```bash
streamlit run Dashboard/app.py
```

## Notes

- The processed files are already present in `Data/processed/`, so the FastAPI app can run immediately.
- Some notebook cells currently use absolute Windows paths. If you rerun notebooks on another machine, update those paths to your local project location or convert them to relative paths.
