# Pricing Engine

FastAPI application and notebook workflow for food-demand pricing analysis. The project prepares a weekly meal-demand dataset, estimates meal-level price elasticity, and generates revenue-oriented price recommendations through an interactive web interface and JSON API.

## Why This Project

This repo turns exploratory pricing work into something closer to a deployable product:

- notebook-driven analysis for data preparation and pricing logic
- a FastAPI web app for business-friendly exploration
- recommendation logic that balances recent demand, historical price bounds, and elasticity
- automated tests plus CI/lint checks for the production code path

## Core Features

- interactive dataset overview with category, cuisine, and week filters
- meal-level price recommendations
- price-response curve inspection for a selected meal
- downloadable recommendation CSV
- JSON API endpoint for recommendation output
- notebook workflow for preparation, EDA, elasticity, and optimization

## Project Structure

```text
.
|-- WebApp/
|   |-- main.py
|   |-- service.py
|   |-- static/
|   `-- templates/
|-- api/
|   `-- index.py
|-- Notebooks/
|   |-- 01_data_preparation.ipynb
|   |-- 02_eda.ipynb
|   |-- 05_price_elasticity.ipynb
|   `-- 06_price_optimization.ipynb
|-- tests/
|   `-- test_service.py
|-- render.yaml
|-- pyproject.toml
|-- requirements.txt
`-- TODO.md
```

## Dataset Source

This project uses the Kaggle Food Demand Forecasting dataset:

`https://www.kaggle.com/datasets/arashnic/food-demand`

Download the dataset manually and place these files in `Data/raw/`:

- `train.csv`
- `meal_info.csv`
- `fulfilment_center_info.csv`

Optional Kaggle files such as `test_QoiMO9B.csv` and `sample_submission_hSlSoT6.csv` are not required by the application.

## Dataset Grain

The main table is `Data/raw/train.csv`.

One row represents:

- one `meal_id`
- in one `center_id`
- for one `week`

Current dataset summary:

- `456,548` observations
- `51` unique meals
- `77` fulfillment centers
- `145` weeks

## Method Overview

The current pricing workflow follows these steps:

1. Merge order history with meal and fulfillment-center metadata.
2. Build a processed dataset with one-hot encoded categorical features and a `revenue` column.
3. Aggregate demand and average price at the `meal_id x week` level.
4. Estimate meal-level elasticity from price and order variation.
5. Use recent weekly demand as the operating baseline.
6. Search candidate prices inside historical and change-based bounds.
7. Select the price with the highest expected revenue for each meal.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create local data folders if needed:

```bash
mkdir -p Data/raw Data/processed
```

Then add the Kaggle CSV files to `Data/raw/`.

## Run The App

```bash
uvicorn WebApp.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API Example

Get recommendations as JSON:

```bash
curl "http://127.0.0.1:8000/api/recommendations?category=Beverages&recent_weeks=4&max_change_pct=20&candidate_count=20"
```

Download recommendations as CSV:

```bash
curl -OJ "http://127.0.0.1:8000/recommendations.csv?category=Beverages&recent_weeks=4"
```

## Notebook Workflow

- `01_data_preparation.ipynb`
  Builds the processed dataset from the raw Kaggle files.
- `02_eda.ipynb`
  Explores pricing and demand patterns in the processed dataset.
- `05_price_elasticity.ipynb`
  Computes meal-level elasticity aggregates.
- `06_price_optimization.ipynb`
  Produces revenue-oriented recommendation candidates.

The notebooks now use repo-relative `Data/raw/` and `Data/processed/` paths, so they can run on another machine without editing hard-coded local paths.

## Quality Checks

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

GitHub Actions runs both checks on pushes and pull requests.

## Deploy On Render

The repo includes `render.yaml` for a one-service Render deployment.

Current service settings:

- runtime: `python`
- plan: `free`
- build command: `pip install -r requirements.txt`
- start command: `uvicorn WebApp.main:app --host 0.0.0.0 --port $PORT`
- health check path: `/health`
- Python version: `3.12.0`

## Notes

- The raw dataset is intentionally not committed to Git.
- The FastAPI app only requires the raw CSV files in `Data/raw/`.
- If `Data/processed/avg_elasticity_per_meal.csv` is missing, the app rebuilds that cache from the raw training data at runtime.
- `requirements.txt` is a thin production install wrapper; dependency definitions live in `pyproject.toml`.
- The repository is FastAPI-only; the previous Streamlit dashboard has been removed.
