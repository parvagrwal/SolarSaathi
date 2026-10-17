# SolarSaathi v2 — Rooftop Solar Advisor

**Internship:** AICTE IBM SkillsBuild ML + Applied AI (CSR Box / BharatCares)
**Deliverable 2 — Final Project (solo)**
**SDGs:** 7 Affordable & Clean Energy (primary) + 13 Climate Action (secondary)

## Problem
Many Indian homeowners want rooftop solar but get vague estimates from installers.
They do not know: how much power will MY roof make, how much will I save, when will it pay back, and is it worth it?

## Solution
Enter city, roof area, monthly bill, panel type, shade, and roof direction. SolarSaathi v2 predicts:
- Monthly generation (kWh) + confidence range ± MAE (39 kWh)
- Monthly savings (₹)
- Recommended system size (kW, sized to your bill, capped by roof)
- Cost after PM Surya Ghar subsidy (panel-specific price: Mono ₹60k/kW, Poly ₹52k/kW, Thin-film ₹45k/kW)
- Payback period (years)
- CO2 saved per year (kg)
- NEW: side-by-side comparison of all 3 panels (size, roof area needed, generation, savings, cost, payback)
- NEW: 12-month seasonal generation chart (monsoon dip Jun–Aug)
- NEW: Copy Summary button for PPT / form

Input -> AI -> Output:
`city + roof area + bill + panel type` -> ML model predicts generation -> app computes savings / size / cost / payback / CO2.

## How to run (no server needed)
1. Open `index.html` in any browser (double-click).
2. Change city / roof / bill / panel, press **Predict my solar savings**.

To retrain the model:
```
pip install scikit-learn pandas numpy
python3 train.py
```
This regenerates `data/solar_training_data.csv`, `metrics.json`, `model.json`.

## Model + data (honest notes)
- Model: Ridge regression on physics-engineered features (products of area × irradiance × efficiency, etc.).
- Test results on 500 held-out rows: **R2 = 0.9953, MAE = 39.0 kWh** (mean generation 1,503 kWh). See `metrics.json`.
- Dataset: **2,500 rows, physics-based SYNTHETIC** — NOT real meter data. Formula:
  `monthly_kwh = usable_area × panel_efficiency × irradiance × 30 × 0.78 × temp_factor + 3% noise`,
  where usable_area = 70% of roof, temp_factor = 1 − 0.4% per °C above 25°C.
  City irradiance / temperature / tariff are annual-average approximations for 10 Indian cities.
- The web app runs the SAME trained coefficients (exported in `model.json`, embedded in `index.html`).
- Assumptions shown in the app: panel price Mono ₹60,000/kW, Poly ₹52,000/kW, Thin-film ₹45,000/kW; PM Surya Ghar subsidy (₹30,000/kW up to 2 kW, ₹18,000 for 3rd kW, cap ₹78,000), export rate ₹3/kWh, grid factor 0.82 kg CO2/kWh. Shade (0–60%) and direction (South 1.00, Flat 0.97, East/West 0.93, North 0.78) scale generation. Seasonal chart uses fixed India factors (May 1.12 peak, Jul/Aug 0.85 monsoon dip) — illustrative, not city-measured.
- Limitation: this is a learning/demo estimate. A real site survey (shade, roof angle, wiring, net-metering rules) is needed before buying.

## Example (verified in screenshot)
Delhi, 100 sq m roof, ₹3,000/month bill, Mono panel, 10% shade, South-facing (v2 defaults):
- ~385 kWh/month (±39), 3.7 kW system, ₹3,029/month savings, ₹1,25,500 after subsidy, 3.5-year payback, 3,783 kg CO2/year.
- Panel compare on same inputs: Mono 3.7 kW / ₹1,25,500 / 3.5 yr, Poly ~same size with lower price, Thin-film needs most roof area. See screenshots.

## Files
- `index.html` — working web app (open in browser)
- `train.py` — dataset generation + training + evaluation
- `data/solar_training_data.csv` — 2,500-row training data
- `model.json` / `metrics.json` — exported model + test metrics
- `screenshots/` — 01 desktop Delhi, 02 Jaipur example, 03 mobile

## Unfair advantage / why this one
India-specific: city irradiance + tariffs + PM Surya Ghar subsidy logic built in, and the system is sized to the household's actual bill — not a generic "solar is good" calculator. Most online calculators do not combine an ML prediction with subsidy + payback in one step.
