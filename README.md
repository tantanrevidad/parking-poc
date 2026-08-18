# Smart Parking Management System — Proof of Concept (POC)

An enterprise-grade Streamlit application demonstrating every layer of the Smart Parking system end-to-end, featuring predictive availability forecasting, confidence-weighted license plate matching, real-time slot state tracking, and computer vision feasibility validation across **Megaworld Townships** (Uptown Bonifacio & Eastwood City).

---

## 🚀 Key Highlights & Current State

- **Multi-Township Simulation:** Live multi-zone simulation across **Uptown Bonifacio** and **Eastwood City** featuring heterogeneous zone profiles (**Office**, **Mall**, **Residential**) with distinct diurnal curves and weekend behaviors.
- **Grace-Period State Machine:** Real-time 4-state slot tracking (`Free`, `Occupied — Unpaid`, `Occupied — Pending Match`, `Occupied — Likely Vacating Soon`) driven by simulated ticketing logs and grace periods.
- **Confidence-Weighted Fuzzy Matcher:** Real OCR-to-ticket matching algorithm with optical confusable-character scoring ($0 \leftrightarrow O$, $1 \leftrightarrow I$, $8 \leftrightarrow B$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $6 \leftrightarrow G$) and margin enforcement ensuring **0% false positive ticket matches**.
- **ML Occupancy Forecasting Engine:** `scikit-learn` `HistGradientBoostingRegressor` trained on 28 days of 15-minute historical readings using causal feature engineering, evaluated on a chronological holdout dataset against a naive baseline.
- **Dual Computer Vision ALPR Validation:**
  1. **🇵🇭 Philippine CCTV Parking Lot & Gate Dataset:** 20 real-world surveillance video frames from multi-level decks, boom barriers, and low-light basement checkpoints.
  2. **🌍 Academic ALPR Benchmark (OpenALPR):** 14 curated international benchmark photographs with hand-verified ground truth plates.
- **Enterprise Dark Dashboard:** Modern Slate-dark UI (`#0F172A`), quick simulation clock toolbar (`+15m`, `+1h`, `Reset`), township filtering, and interactive Plotly visual diagnostics.

---

## 📊 What's Real vs. Simulated

| Component | Status | Implementation Details |
|---|---|---|
| **Camera Feed / Slot Occupancy** | **Simulated** | Synthetic 28-day 15-minute time series generated with zone-specific mathematical curves + Gaussian noise + event multipliers. |
| **License Plate OCR (App Live Stream)** | **Simulated** | Synthetic reads corrupted via an optical confusion matrix ($0/O, 1/I, 8/B, 5/S, 2/Z, 6/G$) with randomized confidence scores. |
| **Ticketing POS Integration** | **Simulated** | SQLite relational transaction log (`ticketing_records`) tracking entry, payment timestamp, and ticket status. |
| **Plate-to-Ticket Matching Algorithm** | **Real** | Confidence-weighted Levenshtein matching with confusable character penalty discounts in `matcher.py`. |
| **Slot State Machine Engine** | **Real** | State transition rules in `state_machine.py` modeling Free, Unpaid, Pending Match, and Vacating states. |
| **ML Predictive Forecaster** | **Real** | `HistGradientBoostingRegressor` in `predictor.py` trained with time-based holdout validation, MAE metrics, and permutation feature importance. |
| **Computer Vision ALPR Pipeline** | **Real** | YOLOv8n vehicle detection + Sobel-X vertical edge plate localization + OCR + real candidate matching in `cv_demo.py`. |

---

## 🛠️ Quickstart & Setup

### 1. Prerequisites
- **Python 3.10+** (Python 3.10–3.14 supported)
- **Tesseract OCR engine** (optional, for running local CV OCR):
  - *Ubuntu/Debian:* `sudo apt-get install tesseract-ocr`
  - *macOS:* `brew install tesseract`
  - *Windows:* [UB-Mannheim Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Installation
```bash
git clone https://github.com/tantanrevidad/parking-poc.git
cd parking-poc
pip install -r requirements.txt
```

### 3. Generate Database & Launch App
```bash
# Generate data/parking.db (automatically initialized on first app launch if missing)
python generate_data.py

# Launch the Streamlit dashboard
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 👁️ Computer Vision Feasibility Demo

The CV demo evaluates vehicle detection and plate recognition against **real-world images with hand-verified ground truth**:

```bash
# Download Philippine CCTV surveillance dataset (20 frames)
python fetch_ph_dataset.py

# Download Academic OpenALPR benchmark dataset (14 images)
python fetch_real_dataset.py

# Run the CV detection & matching pipeline on all datasets
python cv_demo.py --dataset all
```

> **Privacy & Licensing Note:** Real dataset frames are downloaded fresh from open-access sources and are gitignored to comply with third-party licensing and data privacy guidelines.

---

## 📱 Application Modules & Tabs

1. **Occupancy Map:** Live seat-map grid across Uptown Bonifacio and Eastwood City zones (Basement 1, Level 2, Podium 3, etc.) displaying real-time occupancy, available bays, and slot status color-coding.
2. **Availability Forecast:** Interactive trip planner allowing users to pick any zone and arrival horizon (0 to 12 hours ahead) to receive an ML forecast, baseline comparison, conservative safety margin, and historical trend curve.
3. **Model Performance:** Diagnostic dashboard detailing Model MAE, Baseline MAE, Error Reduction %, Permutation Feature Importance bar chart, and Actual vs. Predicted time-series charts.
4. **Plate Matching:** Interactive slot inspector testing the fuzzy matcher on noisy OCR plate reads, displaying character confidence bars, candidate rankings, and match margin confirmation.
5. **CV Demo:** Dual-dataset visual gallery allowing users to toggle between the **Philippine Parking Lot Dataset** and the **OpenALPR Benchmark**, inspecting YOLOv8 vehicle boxes, localized plate crops, OCR reads, and matcher resolutions.

---

## 🏢 Zone Archetypes

- **🏢 Office Zones:** Morning surge (08:00–09:30), slight midday dip (12:00–13:00), evening drain after 17:30. Minimal weekend volume.
- **🛒 Mall Zones:** Gradual morning volume, sustained afternoon build, peak evening traffic (18:00–21:30), +35% higher traffic on weekends.
- **🏠 Residential Zones:** High overnight occupancy (85–95%), workday drop (08:00–17:00), consistent profile across weekdays and weekends.

---

## 📂 Project Structure

```
parking-poc/
├── .streamlit/
│   └── config.toml               # Custom enterprise dark theme config
├── cv-demo/                      # Local CV outputs (gitignored)
├── docs/
│   ├── Revised_Prototype_Plan.md # Prototype architectural roadmap
│   ├── SYSTEM_DOCUMENTATION.md   # Comprehensive technical specification
│   ├── Smart_Parking_Implementation_Plan.pdf
│   └── Smart_Parking_POC_Technical_Implementation_Plan.pdf
├── app.py                        # Streamlit Enterprise Dashboard
├── cv_demo.py                    # Computer Vision & ALPR evaluation pipeline
├── fetch_ph_dataset.py           # CCTV surveillance dataset extractor
├── fetch_real_dataset.py         # OpenALPR benchmark dataset downloader
├── generate_data.py              # Synthetic database & time-series generator
├── matcher.py                    # Confidence-weighted fuzzy matching algorithm
├── predictor.py                  # ML forecasting engine (HistGradientBoosting)
├── requirements.txt              # Project dependencies
├── simulate.py                   # Real-time state simulation helper
├── state_machine.py              # Parking slot lifecycle state machine
└── yolov8n.pt                    # Pre-trained YOLOv8 vehicle detector model
```matching + graceful status states + CV
feasibility) actually working.