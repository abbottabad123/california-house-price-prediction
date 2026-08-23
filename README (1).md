# California House Price Prediction

A regression project that predicts median house value for California block groups, using real 1990 census data. Includes a trained model and a browser-based calculator that runs the model live.

## Files

| File | What it is |
|---|---|
| `housing.csv` | Raw dataset — 20,640 California block groups (rows), 10 columns |
| `train_house_price.py` | Trains and evaluates the models, prints metrics, exports `model_export.json` |
| `model_export.json` | Trained model weights, feature ranges, and accuracy metrics |
| `house_price_calculator.html` | Interactive calculator — open it in any browser, no install needed |

## Dataset

Source: 1990 U.S. Census, California. Each row is one block group (roughly 600–3,000 people).

**Features used:**
- `longitude`, `latitude` — location
- `housing_median_age` — median age of houses in the block group
- `total_rooms`, `total_bedrooms` — totals across all households in the block group
- `population`, `households`
- `median_income` — in tens of thousands of dollars
- `ocean_proximity` — categorical: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`

**Target:** `median_house_value` (in USD, capped at $500,001 in the source data)

## Models trained

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.649 | $50,413 | $69,298 |
| Random Forest (200 trees) | 0.818 | $32,874 | $49,947 |

Random Forest is the more accurate model. Linear Regression is less accurate but its weights are simple enough to run instantly in a browser with no server — that's what powers the calculator.

**What matters most for price** (Random Forest feature importance):
1. Median income (~50%)
2. Ocean proximity — inland vs coastal (~15%)
3. Longitude / latitude (~21% combined)
4. House age, population, rooms, bedrooms, households (~14% combined)

## How to run the training script

```bash
pip install scikit-learn pandas numpy
python3 train_house_price.py
```

This reads `housing.csv`, trains both models, prints metrics to the console, and writes `model_export.json`.

## How to use the calculator

Open `house_price_calculator.html` directly in a browser (double-click it, or drag it into a browser window). Move the sliders or change the location dropdown — the predicted price updates instantly. It's running the trained Linear Regression weights in JavaScript, so it works fully offline.

## Limitations

- Data is from **1990** — not adjusted for inflation or current market conditions.
- Target values are capped at $500,001, so predictions near or above that are extrapolation.
- `ISLAND` category has only 5 examples in the dataset, so predictions for it are unreliable.
- The calculator uses Linear Regression (R² 0.649), not the more accurate Random Forest (R² 0.818), because tree-based models don't translate to a simple formula that can run in a browser.

## Next steps (optional)

- Swap in a different / more recent dataset (e.g. your own city's listings) — just replace `housing.csv` with the same column names and rerun the script.
- Deploy the Random Forest model behind a small API (Flask/FastAPI) for more accurate live predictions.
- Add more features (school ratings, crime data, square footage) if available.
