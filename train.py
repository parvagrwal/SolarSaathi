"""
SolarSaathi — Rooftop Solar Advisor
SDG 7 (Affordable & Clean Energy) + SDG 13 (Climate Action)

What this does:
- Generates a PHYSICS-BASED synthetic training dataset (2,500 rows).
  Formula: monthly_kwh = usable_area * panel_efficiency * irradiance
             * 30 days * performance_ratio * temp_factor + noise
  Irradiance / temperature are annual-average values for 10 Indian cities
  (public climatology approximations, listed in CITIES).
  This is labelled synthetic throughout — NOT claimed as real meter data.
- Trains a Ridge regression model on engineered features.
- Evaluates on a held-out test set, saves metrics + exported coefficients
  (model.json) so the static web app can run the SAME model in the browser.

Run: python3 train.py
"""
import json, math, random
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

random.seed(42); np.random.seed(42)

CITIES = {
    # city: (irradiance kWh/m2/day, avg temp C, tariff Rs/kWh)
    "Delhi": (5.0, 32.0, 8.0),
    "Mumbai": (4.9, 30.0, 9.5),
    "Jaipur": (5.8, 31.0, 7.5),
    "Ahmedabad": (5.6, 32.5, 7.0),
    "Bengaluru": (5.2, 27.0, 8.5),
    "Chennai": (5.1, 31.0, 9.0),
    "Hyderabad": (5.3, 30.0, 8.0),
    "Kolkata": (4.7, 30.5, 8.5),
    "Pune": (5.4, 28.0, 8.5),
    "Lucknow": (5.0, 31.0, 7.5),
}
PANELS = {"Mono (21%)": 0.21, "Poly (17%)": 0.17, "Thin-film (12%)": 0.12}

def make_row():
    city = random.choice(list(CITIES.keys()))
    irr, temp, tariff = CITIES[city]
    # small realistic variation around city averages
    irr_i = max(3.5, irr + np.random.normal(0, 0.3))
    temp_i = temp + np.random.normal(0, 2.0)
    roof_area = round(random.uniform(20, 200), 1)
    usable = roof_area * 0.70  # 30% lost to shade / walkways / tanks
    panel = random.choice(list(PANELS.keys()))
    eff = PANELS[panel]
    pr = 0.78  # performance ratio (inverter + wiring + soiling)
    temp_factor = 1 - 0.004 * max(0, temp_i - 25)  # -0.4%/C above 25C
    monthly_kwh = usable * eff * irr_i * 30 * pr * temp_factor
    monthly_kwh *= (1 + np.random.normal(0, 0.03))  # 3% noise
    monthly_bill = round(random.uniform(500, 12000), 0)
    return {
        "city": city, "roof_area_sqm": roof_area, "usable_area_sqm": round(usable, 1),
        "panel_type": panel, "panel_efficiency": eff,
        "irradiance": round(irr_i, 2), "temperature_c": round(temp_i, 1),
        "monthly_bill_rs": monthly_bill, "tariff_rs_per_kwh": tariff,
        "monthly_kwh": round(max(0, monthly_kwh), 1),
    }

def features(df):
    # Engineered features that mirror the physics (products of inputs)
    return pd.DataFrame({
        "area_x_irr": df["usable_area_sqm"] * df["irradiance"],
        "area_x_eff": df["usable_area_sqm"] * df["panel_efficiency"],
        "area_irr_eff": df["usable_area_sqm"] * df["irradiance"] * df["panel_efficiency"],
        "area_irr_temp": df["usable_area_sqm"] * df["irradiance"] * df["temperature_c"],
        "irradiance": df["irradiance"],
        "temperature_c": df["temperature_c"],
        "usable_area": df["usable_area_sqm"],
    })

if __name__ == "__main__":
    rows = [make_row() for _ in range(2500)]
    df = pd.DataFrame(rows)
    df.to_csv("data/solar_training_data.csv", index=False)
    print(f"Dataset: {len(df)} rows -> data/solar_training_data.csv")
    print(df.head(3).to_string(index=False))

    X = features(df); y = df["monthly_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": "Ridge regression on physics-engineered features",
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "r2_test": round(float(r2_score(y_test, pred)), 4),
        "mae_kwh_test": round(float(mean_absolute_error(y_test, pred)), 2),
        "mean_monthly_kwh": round(float(y.mean()), 1),
        "dataset": "Physics-based synthetic, 2500 rows, 10 Indian cities. NOT real meter data.",
        "cities": CITIES,
        "panels": PANELS,
    }
    print("\nMetrics:", json.dumps(metrics, indent=2))
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    exported = {
        "feature_order": list(X.columns),
        "coefficients": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "cities": CITIES, "panels": PANELS,
    }
    with open("model.json", "w") as f:
        json.dump(exported, f, indent=2)
    print("Saved metrics.json + model.json")
    # Sanity check: Delhi, 100 sqm roof, mono panel
    sample = pd.DataFrame([{"usable_area_sqm": 70, "irradiance": 5.0, "temperature_c": 32.0, "panel_efficiency": 0.21}])
    print("Sanity check Delhi 100sqm mono ->", round(float(model.predict(features(sample))[0]), 1), "kWh/month")
