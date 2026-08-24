# Smart Parking Management System — Proof of Concept (POC)

An enterprise-grade Streamlit application demonstrating every layer of the Smart Parking system end-to-end, featuring predictive availability forecasting, confidence-weighted license plate matching, real-time slot state tracking, interactive direct-click SQLite database telemetry inspection, and computer vision feasibility validation across **Megaworld Townships** (Uptown Bonifacio & Eastwood City).

---

## 🚀 Key Highlights & Current State

- **Township Deck Layout & Sequential Ordering:** Realistic township architectural scheme organized sequentially by commercial archetype (**Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**) with labeled Drive Aisles (Lane A, Lane B, Lane C) and expanded capacity (48 Mall bays, 24 Office bays, 16 Residential bays).
- **Zero-Page-Reload Direct Stall Inspection:** Every parking stall block is an interactive solid-color card powered by isolated fragment execution (`@st.fragment`). Clicking directly on any block instantly opens the **SQLite Database Row Inspector modal (`@st.dialog`)** with zero page reload or scroll jitter, displaying real-time telemetry, tickets, OCR reads, and raw SQL queries.
- **Dynamic Real-Time PST Hero Clock (Tab 1):** Live client-side Philippine Standard Time clock counting up seconds, minutes, and hours (`HH:MM:SS AM/PM`) dynamically with zero server overhead and a pulsing live monitoring badge.
- **Predictive Timeline Control (Tab 2):** Dedicated future simulation interface with calendar date selection, 12-hour/minute pickers, direct time entry (`14:30`, `2:30pm`, `09:00`), and quick-step toolbar (`-1h`, `-15m`, `+15m`, `+1h`, `Live`).
- **Stabilized Simulation State Machine:** Real-time 4-state slot tracking (`Available (Free)`, `Occupied — Unpaid`, `Occupied — Pending Match`, `Occupied — Likely Vacating Soon`) driven by simulated ticketing logs, grace periods, and stabilized 5-minute simulation window seeds.
- **Confidence-Weighted Fuzzy Matcher:** Real OCR-to-ticket matching algorithm with optical confusable-character scoring ($0 \leftrightarrow O$, $1 \leftrightarrow I$, $8 \leftrightarrow B$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $6 \leftrightarrow G$) and margin enforcement ensuring **0% false positive ticket matches**.
- **ML Occupancy Forecasting Engine:** `scikit-learn` `HistGradientBoostingRegressor` trained on 28 days of 15-minute historical readings using causal feature engineering, evaluated on a chronological holdout dataset against a naive baseline.
- **Dual Computer Vision ALPR Validation:**
  1. **Philippine CCTV Parking Lot & Gate Dataset:** 20 real-world surveillance video frames from multi-level decks, boom barriers, and low-light basement checkpoints.
  2. **Academic ALPR Benchmark (OpenALPR):** 14 curated international benchmark photographs with hand-verified ground truth plates.
- **5-Phase Computer Vision Space Detection Engine:** YOLOv8n vehicle detector with adaptive low-light CLAHE contrast enhancement, true perspective polygon ROI calibration (`slots_config.json`), Intersection over Area (IoA) occupancy scoring with centroid containment, and 5-frame temporal state debouncing.
- **High-Contrast Dark & Light Theme Engine:** Segmented theme switcher with responsive styling, bold pure-black tab typography in Light Mode, and adaptive Plotly chart themes.

---

## 📊 What's Real vs. Simulated

| Component | Status | Implementation Details |
|---|---|---|
| **Camera Feed / Slot Occupancy** | **Simulated** | Synthetic 28-day 15-minute time series generated with zone-specific mathematical curves + Gaussian noise + event multipliers. |
| **License Plate OCR (App Live Stream)** | **Simulated** | Synthetic reads corrupted via an optical confusion matrix ($0/O, 1/I, 8/B, 5/S, 2/Z, 6/G$) with randomized confidence scores. |
| **Ticketing POS Integration** | **Simulated** | SQLite relational transaction log (`ticketing_records` in `data/parking.db`) tracking entry, payment timestamp, and ticket status. |
| **Plate-to-Ticket Matching Algorithm** | **Real** | Confidence-weighted Levenshtein matching with confusable character penalty discounts in `matcher.py`. |
| **Slot State Machine Engine** | **Real** | State transition rules in `state_machine.py` modeling Free, Unpaid, Pending Match, and Vacating states. |
| **ML Predictive Forecaster** | **Real** | `HistGradientBoostingRegressor` in `predictor.py` trained with time-based holdout validation, MAE metrics, and permutation feature importance. |
| **Computer Vision ALPR Pipeline** | **Real** | YOLOv8n vehicle detection + Sobel-X vertical edge plate localization + OCR + real candidate matching in `cv_demo.py`. |
| **Computer Vision Space Detector** | **Real** | 5-phase ROI & IoA vehicle detection engine with CLAHE enhancement in `parking_detector.py`. |

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

## 🗄️ Inspecting SQLite Database Records

The persistent relational database is stored locally in `data/parking.db`.

### Method A: In-Dashboard Direct Click Inspection
1. Navigate to **Tab 1: Occupancy Map**.
2. Click **directly on any colored parking stall block** (`M-001`, `O-001`, `R-001`, etc.).
3. An interactive modal dialog opens instantly (with zero page reload via `@st.fragment`) displaying:
   - **Active Live Telemetry** — Consolidated snapshot of state, ticket, and OCR reading.
   - **Active Occupancy State (`current_state`)** — Current status and state change timestamp.
   - **Ticketing & Settlement (`ticketing_records`)** — Assigned vehicle plate, entry time, payment settlement time.
   - **Optical Plate Read (`plate_reads`)** — Camera OCR detected text, confidence vector, ground truth plate.
   - **Infrastructure Metadata (`slots` & `zones`)** — Deck level, archetype, township site, zone capacity.
   - **Executed SQLite Statement (Raw SQL)** — Underlying parameterized SQL statement executed against `data/parking.db`.

### Method B: External Database Viewers
You can open `data/parking.db` directly with any SQLite GUI tool:
- **DB Browser for SQLite:** [sqlitebrowser.org](https://sqlitebrowser.org/)
- **VS Code / Cursor Extensions:** `SQLite Viewer` or `Database Client`
- **CLI Query:**
  ```powershell
  python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('data/parking.db'); print(pd.read_sql('SELECT * FROM current_state LIMIT 10', conn))"
  ```

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

1. **Occupancy Map:** Live real-time operations deck displaying Uptown Bonifacio and Eastwood City zones ordered sequentially by archetype (**Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**). Displays live capacity KPIs, Drive Aisles, a dynamic counting PST clock, and solid-color clickable blocks (🟢 Available, 🔴 Occupied, 🟡 Pending Match, 🔵 Vacating) that trigger in-place SQLite modal inspection with zero page reload.
2. **Availability Forecast:** Dedicated future simulation & predictive planning tool allowing users to pick any zone, calendar date, and arrival time (using 12-hour dropdowns, minute pickers, or direct text time inputs) to receive an ML forecast, baseline comparison, conservative safety margin, and historical trend curve.
3. **Model Performance:** Diagnostic dashboard detailing Model MAE, Baseline MAE, Error Reduction %, Permutation Feature Importance bar chart, and Actual vs. Predicted time-series charts.
4. **Plate Matching:** Interactive slot inspector testing the fuzzy matcher on noisy OCR plate reads, displaying character confidence bars, candidate rankings, and match margin confirmation.
5. **ALPR Feasibility (CV):** Dual-dataset visual gallery allowing users to toggle between the **Philippine Parking Lot Dataset** and the **OpenALPR Benchmark**, inspecting YOLOv8 vehicle boxes, localized plate crops, OCR reads, and matcher resolutions.
6. **Space Detection (CV):** Enterprise computer vision parking space occupancy detection engine across surveillance feeds in `car_dataset/`, implementing a **5-Phase Occupancy Detection Algorithm**: (1) Dual Native ROI Calibration from `slots_config.json`, (2) YOLOv8n vehicle inference with Adaptive Low-Light CLAHE Boost, (3) Intersection over Area (IoA) spatial occupancy calculation with Centroid Containment, (4) temporal sliding-window state debouncing (5-frame history, $\ge 60\%$ consensus), and (5) standardized JSON output payload generation with real-time overlays, capacity KPIs, and telemetry.

---

## 🏢 Zone Archetypes

- **Office Zones:** Morning surge (08:00–09:30), slight midday dip (12:00–13:00), evening drain after 17:30. Minimal weekend volume.
- **Mall Zones:** Gradual morning volume, sustained afternoon build, peak evening traffic (18:00–21:30), +35% higher traffic on weekends.
- **Residential Zones:** High overnight occupancy (85–95%), workday drop (08:00–17:00), consistent profile across weekdays and weekends.

---

## 📂 Project Structure

```
parking-poc/
├── .streamlit/
│   └── config.toml               # Custom enterprise theme configuration
├── car_dataset/                  # Multi-angle CCTV & parking row surveillance frames
├── cv-demo/                      # Local ALPR benchmark outputs (gitignored)
├── data/
│   └── parking.db                # SQLite database with relational schemas & telemetry
├── debug_output/                 # Visual regression layers & JSON telemetry outputs
├── docs/
│   ├── Revised_Prototype_Plan.md      # Prototype architectural roadmap
│   ├── SYSTEM_DOCUMENTATION.md        # Comprehensive technical specification
│   ├── TAB6_SPACE_DETECTION_GUIDE.md  # Space Detection engine guide
│   ├── Smart_Parking_Implementation_Plan.pdf
│   └── Smart_Parking_POC_Technical_Implementation_Plan.pdf
├── scripts/
│   ├── calibrate_roi.py          # Interactive GUI parking slot ROI calibrator
│   └── debug_overlay.py          # 3-layer visual regression & diagnostic framework
├── app.py                        # Streamlit Enterprise Dashboard (6 Tabs)
├── calibrate.html                # Interactive browser-based polygon annotation tool
├── cv_demo.py                    # Computer Vision & ALPR evaluation pipeline
├── fetch_ph_dataset.py           # CCTV surveillance dataset extractor
├── fetch_real_dataset.py         # OpenALPR benchmark dataset downloader
├── generate_data.py              # Synthetic database & time-series generator
├── matcher.py                    # Confidence-weighted fuzzy matching algorithm
├── parking_detector.py           # 5-phase space occupancy & vehicle detection engine
├── predictor.py                  # ML forecasting engine (HistGradientBoosting)
├── requirements.txt              # Project dependencies
├── simulate.py                   # Real-time state simulation helper
├── slots_config.json             # Calibrated slot ROI polygon configurations
├── state_machine.py              # Parking slot lifecycle state machine
└── yolov8n.pt                    # Pre-trained YOLOv8 vehicle detector model
```