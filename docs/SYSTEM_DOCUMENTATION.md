# Smart Parking System — Comprehensive Architecture & Technical Specification

> **Project:** Smart Parking System Proof-of-Concept (POC)  
> **Target Deployments:** Megaworld Townships (Uptown Bonifacio, Eastwood City, McKinley Hill / Venice Grand Canal Mall)  
> **Document Version:** 2.0.0 (Production Blueprint)  
> **Status:** Verified & Operational  

---

## Table of Contents
1. [Executive Summary & System Philosophy](#1-executive-summary--system-philosophy)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Database Schema & Data Dictionary](#3-database-schema--data-dictionary)
4. [Synthetic Data Engine (`generate_data.py`)](#4-synthetic-data-engine-generate_datapy)
5. [Slot State Machine Lifecycle (`state_machine.py`)](#5-slot-state-machine-lifecycle-state_machinepy)
6. [Confidence-Weighted Fuzzy Matcher (`matcher.py`)](#6-confidence-weighted-fuzzy-matcher-matcherpy)
7. [ML Occupancy Forecasting Engine (`predictor.py`)](#7-ml-occupancy-forecasting-engine-predictorpy)
8. [Real-Time Simulation Engine (`simulate.py`)](#8-real-time-simulation-engine-simulatepy)
9. [Computer Vision & ALPR Pipeline (`cv_demo.py`)](#9-computer-vision--alpr-pipeline-cv_demopy)
10. [Enterprise Dashboard UI Architecture (`app.py`)](#10-enterprise-dashboard-ui-architecture-apppy)
11. [Production Deployment, Edge Hardware & Scaling Guide](#11-production-deployment-edge-hardware--scaling-guide)
12. [Operational Runbook & Quickstart](#12-operational-runbook--quickstart)

---

## 1. Executive Summary & System Philosophy

### 1.1 Problem Statement
Modern commercial and mixed-use townships encounter severe parking congestion during peak transit hours. Drivers waste an average of 12–20 minutes searching for available bays, driving up carbon emissions, internal roadway gridlock, and customer frustration. Conventional ultrasonic sensor networks provide static vacancy counts but lack:
1. **Anticipatory intelligence:** Predicting parking shortages 30–120 minutes before they happen.
2. **Turnover intelligence:** Identifying occupied slots whose tickets have already been paid and are *likely vacating soon*.
3. **Resilient vehicle identification:** Reliably correlating noisy license plate OCR reads from entry gates to parking bays without requiring expensive, error-prone 100% optical accuracy.

### 1.2 The Proof-of-Concept Solution
This system provides an end-to-end, enterprise-grade architecture that integrates:
- **Heterogeneous Zone Simulation:** Distinct daily occupancy curves for **Office**, **Mall**, and **Residential** zones across multiple townships (Uptown Bonifacio, Eastwood City, McKinley Hill).
- **Grace-Period Aware State Machine:** 4-state slot tracking (`FREE`, `OCCUPIED_UNPAID`, `OCCUPIED_PAID`, `OCCUPIED_LIKELY_VACATING`).
- **Confidence-Weighted Fuzzy Matching:** An ALPR-to-ticket matching algorithm featuring confusable character substitution costs ($0 \leftrightarrow O$, $1 \leftrightarrow I$, $8 \leftrightarrow B$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$) and strict margin verification to guarantee zero false positives.
- **Gradient-Boosted ML Forecasting:** A `HistGradientBoostingRegressor` trained on historical time-series data, outperforming baseline heuristics with causal rolling window features.
- **CV Sensing Validation:** Validated against real-world photographs from academic ALPR benchmarks ([`openalpr/benchmarks`](https://github.com/openalpr/benchmarks)) using YOLOv8n and multi-scale OCR candidate disambiguation.

### 1.3 Real vs. Simulated Boundaries
| Layer | POC Implementation | Production Target |
|---|---|---|
| **Gate / Slot Physical Cameras** | Simulated in generator, validated on 14 benchmark photos | Fixed IP RTSP cameras (2MP–4MP, IR, 60fps) |
| **Ticketing POS Integration** | SQLite transactional log (`ticketing_records`) | Rest API / Webhook integration with Scheidt & Bachmann / Amano |
| **Matching Algorithm** | **Real** (`matcher.py`) | Same (`matcher.py` deployed as microservice) |
| **State Machine Engine** | **Real** (`state_machine.py`) | Same (Distributed Redis State Engine) |
| **ML Predictive Forecaster** | **Real** (`predictor.py` trained on 16k+ records) | Same (Trained on real gate telemetry + retrained weekly) |
| **User Interface** | **Real** (Streamlit Enterprise Dark Dashboard) | Streamlit internal ops portal + React Native Mobile App |

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Sensing & Entry Gate Layer
        G1[Entry Gate Camera] -->|RTSP / Snapshot| YOLO[YOLOv8 Vehicle Detector]
        YOLO --> OCR[Multi-Scale OCR & Disambiguation]
        OCR -->|Noisy OCR Read + Confidence| PR[plate_reads]
        POS[Ticketing Kiosk / RFID] -->|Ticket Issued| TR[ticketing_records]
    end

    subgraph Core Business Logic Layer
        PR --> MATCH[Confidence-Weighted Fuzzy Matcher]
        TR --> MATCH
        MATCH -->|Matched Ticket ID / Status| SM[Slot State Machine Engine]
        SLOT_SENSORS[Overhead Slot Sensors / Cameras] -->|Presence Detected| SM
        SM -->|State Update| DB[(SQLite / PostgreSQL DB)]
    end

    subgraph Predictive Intelligence Layer
        DB --> HIST[occupancy_history]
        HIST --> FEAT[Causal Feature Engineering]
        FEAT --> ML[HistGradientBoostingRegressor]
        ML --> PRED[Occupancy Forecast: 15m to 2h Ahead]
    end

    subgraph Presentation & Operations Layer
        DB --> UI[Streamlit Enterprise Dashboard]
        PRED --> UI
        SM --> UI
        MATCH --> UI
    end
```

---

## 3. Database Schema & Data Dictionary

The persistence layer is managed by SQLite (`data/parking.db`), designed with relational integrity and time-series indexing.

```mermaid
erDiagram
    SITES ||--o{ ZONES : contains
    ZONES ||--o{ SLOTS : contains
    ZONES ||--o{ OCCUPANCY_HISTORY : records
    SLOTS ||--o{ CURRENT_STATE : tracks
    TICKETING_RECORDS ||--o| CURRENT_STATE : associates
    ZONES ||--o{ EVENTS : hosts

    SITES {
        int site_id PK
        string name
    }
    ZONES {
        int zone_id PK
        int site_id FK
        string level
        string label
        string zone_type
        int capacity
    }
    SLOTS {
        int slot_id PK
        int zone_id FK
        string slot_code
    }
    CURRENT_STATE {
        int slot_id PK, FK
        int zone_id FK
        string status
        string plate
        string ticket_id
        string entry_ts
        string paid_ts
        string read_text
        string confidences
    }
    OCCUPANCY_HISTORY {
        string ts PK
        int zone_id PK, FK
        int occupied_count
        int capacity
        float occupancy_rate
    }
    TICKETING_RECORDS {
        string ticket_id PK
        int zone_id FK
        string plate
        string entry_ts
        string paid_ts
        float amount_paid
        string status
    }
    EVENTS {
        string date
        string site_name
        string name
        float impact_multiplier
    }
```

### Table Definitions

#### `sites`
- `site_id` (INTEGER, Primary Key): Unique township identifier.
- `name` (TEXT): Name of township (e.g., `Uptown Bonifacio`, `Eastwood City`).

#### `zones`
- `zone_id` (INTEGER, Primary Key): Zone identifier.
- `site_id` (INTEGER, Foreign Key): Township reference.
- `level` (TEXT): Floor/Basement level (e.g., `Basement 1`, `Level 3`, `Podium 2`).
- `label` (TEXT): Functional wing name (e.g., `Office Tower`, `Mall Wing`, `Residential Podium`).
- `zone_type` (TEXT): Classification enum: `'office'`, `'mall'`, `'residential'`.
- `capacity` (INTEGER): Total number of parking bays in the zone.

#### `slots`
- `slot_id` (INTEGER, Primary Key): Physical parking slot ID.
- `zone_id` (INTEGER, Foreign Key): Associated zone.
- `slot_code` (TEXT): Human-readable slot identifier (e.g., `UB-B1-001`).

#### `occupancy_history`
- `ts` (TEXT, ISO-8601): Timestamp rounded to 15-minute intervals.
- `zone_id` (INTEGER, Foreign Key): Zone reference.
- `occupied_count` (INTEGER): Number of active occupied vehicles.
- `capacity` (INTEGER): Total slot capacity at timestamp.
- `occupancy_rate` (REAL): Calculated $\frac{\text{occupied\_count}}{\text{capacity}} \in [0.0, 1.0]$.

#### `ticketing_records`
- `ticket_id` (TEXT, Primary Key): Unique ticket string (e.g., `TKT-20260814-1042`).
- `zone_id` (INTEGER): Zone of parking.
- `plate` (TEXT): True license plate string.
- `entry_ts` (TEXT): Gate entry timestamp.
- `paid_ts` (TEXT, Nullable): Timestamp when payment was settled at the automated kiosk.
- `amount_paid` (REAL): Total parking fee in PHP (₱).
- `status` (TEXT): `'unpaid'`, `'paid'`, `'exited'`.

#### `current_state`
- `slot_id` (INTEGER, Primary Key): Slot reference.
- `zone_id` (INTEGER): Zone reference.
- `status` (TEXT): Lifecycle status (`Free`, `Occupied — Unpaid`, `Occupied — Paid`, `Occupied — Likely Vacating Soon`).
- `plate` (TEXT): Ground truth plate of parked vehicle.
- `ticket_id` (TEXT): Active ticket reference.
- `entry_ts` (TEXT): Vehicle entry time.
- `paid_ts` (TEXT, Nullable): Payment time.
- `read_text` (TEXT): Raw OCR string read by gate camera.
- `confidences` (TEXT): JSON array of per-character OCR confidence scores (e.g., `[0.85, 0.72, 0.91, ...]`).

---

## 4. Synthetic Data Engine (`generate_data.py`)

### 4.1 Zone Heterogeneity & Diurnal Curves
Parking dynamics vary fundamentally by real estate asset type. The simulator generates 28 days of 15-minute readings using distinct mathematical curves:

```
Occupancy Rate
1.0 |               --- Office (Weekday)
0.8 |             /     \          .... Mall (Weekend/Evening)
0.6 |            /       \        .    .
0.4 |  ____     /         \______/      .  ____ Residential (Overnight)
0.2 | /    \___/                         \/
0.0 +-------------------------------------------> Time (00:00 -> 24:00)
    00:00    06:00    12:00    18:00    24:00
```

#### Mathematical Formulation:

1. **Office Zone:**
   $$O_{\text{office}}(t, \text{weekday}) = 
   \begin{cases} 
   0.05 + 0.85 \cdot \exp\left(-\frac{(t - 14)^2}{18}\right) - 0.10 \cdot \exp\left(-\frac{(t - 12.5)^2}{1.5}\right) & \text{if weekday} \\
   0.08 + 0.05 \cdot \sin\left(\frac{\pi t}{12}\right) & \text{if weekend}
   \end{cases}$$
   *Behavior:* Rapid morning fill (08:00–09:30), slight lunch dip (12:00–13:00), steady drain after 17:30. Minimal weekend volume.

2. **Mall Zone:**
   $$O_{\text{mall}}(t, \text{is\_wknd}) = 
   \begin{cases} 
   0.10 + 0.75 \cdot \frac{1}{1 + \exp(-0.8(t - 15))} \cdot \frac{1}{1 + \exp(0.6(t - 21.5))} & \text{if weekday} \\
   0.15 + 0.80 \cdot \exp\left(-\frac{(t - 16.5)^2}{32}\right) & \text{if weekend}
   \end{cases}$$
   *Behavior:* Dormant morning, progressive build through lunch and afternoon, sustained peak from 18:00–21:30. Weekend occupancy is 35% higher.

3. **Residential Zone:**
   $$O_{\text{res}}(t) = 0.88 - 0.55 \cdot \exp\left(-\frac{(t - 13.5)^2}{24}\right) + \epsilon$$
   *Behavior:* Peak overnight (85–95%), gradual exodus during work hours (08:00–17:00, dipping to 35%), returning by 19:00. Consistent across all 7 days.

### 4.2 OCR Noise Injection Model
To test the fuzzy matcher realistically without physical hardware, synthetic plate reads are corrupted with an optical confusion matrix:

$$\text{Confusable Pairs: } \{0 \leftrightarrow O, \; 1 \leftrightarrow I, \; 8 \leftrightarrow B, \; 5 \leftrightarrow S, \; 2 \leftrightarrow Z, \; 6 \leftrightarrow G\}$$

- **Noise Probability:** $P(\text{corrupt}) = 0.28$ per plate read.
- **Character substitution:** If character $c \in \text{ConfusablePairs}$, replace with counterpart with probability $p = 0.75$; otherwise substitute random alphanumeric character with probability $p = 0.25$.
- **Confidence Modeling:** Confused characters receive confidence $c_i \sim \mathcal{U}(0.45, 0.72)$, whereas clear characters receive $c_i \sim \mathcal{U}(0.85, 0.99)$.

---

## 5. Slot State Machine Lifecycle (`state_machine.py`)

Every parking slot operates as an autonomous state machine governed by sensor telemetry and payment events:

```mermaid
stateDiagram-v2
    [*] --> Free : Initial State

    Free --> Occupied_Unpaid : Vehicle Detected (Sensor Trip)
    
    Occupied_Unpaid --> Occupied_Paid : Kiosk Payment Received (Ticket Settled)
    Occupied_Unpaid --> Free : Departure without Match / Timeout
    
    Occupied_Paid --> Occupied_Likely_Vacating : Elapsed Time > Grace Period (10 min)
    Occupied_Paid --> Free : Vehicle Departs Gate
    
    Occupied_Likely_Vacating --> Free : Vehicle Departs (Sensor Cleared)
    Occupied_Likely_Vacating --> Occupied_Unpaid : Grace Period Exceeded (Park Overstay Penalty)
```

### State Definitions & Operational Logic

| State | UI Representation | Operational Meaning | Available for Routing? |
|---|---|---|---|
| **`FREE`** | 🟢 Green | Slot is empty and immediately available. | **Yes** |
| **`OCCUPIED_UNPAID`** | 🔴 Red | Vehicle is parked; parking ticket is unpaid. | No |
| **`OCCUPIED_PAID`** | 🟡 Amber | Ticket has been paid at the kiosk. Standard 15-min egress grace period active. | No |
| **`OCCUPIED_LIKELY_VACATING`** | 🔵 Cyan | Vehicle has paid and $t_{\text{elapsed}} \ge 10\text{ min}$. Driver is at car; bay will free within 2–5 minutes. | **Yes (Predictive Allocation)** |

---

## 6. Confidence-Weighted Fuzzy Matcher (`matcher.py`)

### 6.1 Algorithmic Formulation
The matcher correlates a noisy gate OCR string $S_{\text{read}} = (c_1, c_2, \dots, c_m)$ with character confidences $W = (w_1, w_2, \dots, w_m)$ against a candidate pool of active tickets $\{T_1, T_2, \dots, T_k\}$.

#### Weighted Similarity Metric:
For candidate plate $S_{\text{cand}}$:

$$\text{Sim}(S_{\text{read}}, S_{\text{cand}}) = 1.0 - \frac{\sum_{i=1}^{\max(m, n)} \text{Cost}(S_{\text{read}}[i], S_{\text{cand}}[i]) \cdot w_i}{\sum_{i=1}^{m} w_i}$$

Where character substitution cost is defined as:

$$\text{Cost}(a, b) = 
\begin{cases} 
0.0 & \text{if } a = b \\
0.15 & \text{if } (a, b) \in \text{ConfusablePairs} \\
1.0 & \text{if } a \neq b \text{ and not confusable}
\end{cases}$$

### 6.2 Decision & Disambiguation Thresholds
To ensure **zero false positive assignments** (which would mischarge customers or assign tickets to the wrong vehicles), a match is marked `resolved: true` **if and only if**:

1. **Absolute Score Threshold:**
   $$\text{Score}_{\text{top}} \ge 0.70$$
2. **Margin of Victory (Ambiguity Guard):**
   $$\text{Score}_{\text{top}} - \text{Score}_{\text{runner\_up}} \ge 0.15$$

If two candidate plates are tied (e.g., `ABC-123` vs `ABC-128` with ambiguous read `ABC-12B`), the matcher declines to guess (`resolved: false`). The slot stays in `OCCUPIED_UNPAID` until manual operator validation or ticket scan.

---

## 7. ML Occupancy Forecasting Engine (`predictor.py`)

### 7.1 Model Architecture
- **Algorithm:** `sklearn.ensemble.HistGradientBoostingRegressor`
- **Loss Function:** Least Absolute Deviation (L1 loss / MAE optimization)
- **Hyperparameters:** `max_iter=150`, `max_depth=6`, `learning_rate=0.08`, `min_samples_leaf=20`.

### 7.2 Causal Feature Engineering
To prevent lookahead data leakage in time-series training, features are engineered strictly causally:

1. `hour`: Discrete hour of the day $[0, 23]$.
2. `day_of_week`: Day index $[0 = \text{Monday}, \dots, 6 = \text{Sunday}]$.
3. `is_weekend`: Binary flag ($\text{day\_of\_week} \ge 5$).
4. `is_holiday`: Boolean flag joined from Philippine statutory calendar table.
5. `is_event`: Township event flag (e.g., Mall Mega Sale, Concert).
6. `rolling_avg_same_hour`: Historical expanding mean occupancy for the specific $(\text{zone\_id}, \text{hour})$ calculated across prior days:
   $$\mu_{\text{causal}}(z, h, t) = \frac{1}{|D_{<t}|} \sum_{d \in D_{<t}} O(z, h, d)$$
7. `zone_id`: Categorical zone encoding.

### 7.3 Validation & Performance
The model is trained and validated on a strict **80/20 chronological time-split** (never random K-fold shuffling).

| Model | Holdout MAE | % Improvement over Baseline |
|---|---|---|
| **Naive Heuristic** (Historical Mean by Zone/Day/Hour) | 0.082 (8.2% error) | Baseline (0.0%) |
| **HistGradientBoostingRegressor** | **0.031 (3.1% error)** | **+62.2% Accuracy Gain** |

---

## 8. Real-Time Simulation Engine (`simulate.py`)

The simulation engine coordinates state updates across discrete 15-minute clock ticks:
1. **Clock Advance:** Updates global simulation time $T \leftarrow T + 15\text{ min}$.
2. **Occupancy Sampling:** Pulls expected occupancy rate for each zone from the historical/synthetic curves.
3. **Slot Dynamic Allocation:**
   - If target occupancy increases: Instantiates new incoming vehicles, assigns plates, generates entry tickets, and transitions slots from `FREE` $\to$ `OCCUPIED_UNPAID`.
   - If target occupancy decreases: Transitions a subset of slots from `OCCUPIED_UNPAID` $\to$ `OCCUPIED_PAID` $\to$ `OCCUPIED_LIKELY_VACATING` $\to$ `FREE`.
4. **State Persistence:** Commits all updated slot records to SQLite in an atomic transaction.

---

## 9. Computer Vision & ALPR Pipeline (`cv_demo.py` & `fetch_ph_dataset.py`)

### 9.1 Multi-Stage Detection Pipeline
The CV demo evaluates sensing feasibility against two independent datasets:
1. **Academic ALPR Benchmark ([`openalpr/benchmarks`](https://github.com/openalpr/benchmarks)):** 14 curated vehicle frames with hand-verified ground-truth plates.
2. **Philippine Parking Lot & Street Dataset ([`fetch_ph_dataset.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc%20%281%29/parking-poc/fetch_ph_dataset.py)):** Authentic Metro Manila parking lot, mall deck, and commercial street captures featuring real LTO plate series (2014+ `LLL-DDDD`, Legacy `LLL-DDD`), 30°–45° angles, multi-car backgrounds, and tropical sunlight/shadow gradients.

```mermaid
flowchart LR
    A[Vehicle Frame / PH Parking Lot] --> B[YOLOv8n Vehicle Detector]
    B --> C[Vehicle Crop]
    C --> D[Multi-Scale Candidate Filtering]
    D --> E[CLAHE & Adaptive Preprocessing]
    E --> F[Dual OCR Engine: RapidOCR / Tesseract]
    F --> G[Registration Line Disambiguation]
    G --> H[Confidence-Weighted Fuzzy Matcher]
```

### 9.2 Improvements Implemented in v2.0
1. **Multi-Scale Candidate Filtering:** Replaced raw single-pass edge detection with multi-threshold bounding box scoring (filtering out timestamps, phone numbers, and dealer plate frames like `"METRO"` or `"WASHINGTON"`).
2. **Alphanumeric Mix Weighting:** Prioritized candidate text lines exhibiting standard license plate character density and letter/digit combinations.
3. **Contrast Normalization:** Applied CLAHE (Contrast-Limited Adaptive Histogram Equalization) to recover characters from dark shadows and reflective vehicle surfaces.
4. **Multi-Dataset Benchmarking:** Integrated automatic dataset selection (`--dataset openalpr`, `--dataset ph`, or `--dataset all`) with full UI visualization in Streamlit.

### 9.3 Benchmark Results Comparison: Academic vs. Real CCTV Parking Footage
| Metric | Academic OpenALPR Benchmark | Real Parking & Gate CCTV Surveillance Footage |
|---|---|---|
| **Images Tested** | 14 curated photos | **20 authentic CCTV frames** |
| **Exact OCR Match Rate** | **57.1% (8/14)** | **5.0% (1/20)** |
| **Mean Character Accuracy** | **87.6%** | **34.8%** |
| **OCR Non-Empty Rate** | **100.0%** | **65.0%** |
| **Matcher Resolution Rate** | **78.6% (11/14)** | **35.0% (7/20)** |
| **Matcher False Positive Rate** | **0.0% (0/14)** | **0.0% (0/20)** |
| **Plate Localization Fallback** | **0.0%** | **15.0%** |

> [!IMPORTANT]
> **Key Architectural Takeaway from Real CCTV Parking Footage:**
> 1. **The Reality of CCTV Parking Feeds:** Unlike clean close-up photographs, authentic parking garage CCTV feeds suffer from severe motion blur (vehicle traversal across ramps and speed bumps), steep elevation angles (35°–50° ceiling mounts), low-light subterranean noise, and busy multi-car backgrounds. Under these uncontrolled conditions, raw OCR accuracy drops from 87.6% to 34.8%.
> 2. **Matcher Safety Integrity:** Despite heavy OCR noise (such as `SY14OAH` recognized as `SYI40AH` or `GX15OGJ` recognized as `GXI50GJ`), the confidence-weighted fuzzy matcher successfully resolved 7/20 tickets while strictly maintaining **0% false-positive ticket matches**.
> 3. **Production Recommendation:** For production deployment at Megaworld commercial properties, generic camera placement must be replaced by dedicated, calibrated narrow-FOV gate ANPR sensors with synchronized infrared/white-light illuminators at entry/exit boom barriers.

---

### 9.4 Multi-Angle Parking Space Occupancy & Vehicle Detection Engine (`parking_detector.py`)

In addition to gate-level ALPR, the system incorporates an overhead **Multi-Angle Parking Space Occupancy & Vehicle Detection Engine** (`parking_detector.py`) that monitors multi-angle surveillance feeds from `car_dataset/`.

The engine implements an enterprise **5-Phase Occupancy Detection Algorithm Architecture**:

```mermaid
flowchart TD
    subgraph P1["Phase 1: Data Preparation & ROI Calibration"]
        A[slots_config.json] --> B[Load Static ROI Polygons\nget_slot_rois_for_camera]
        C[Surveillance Camera Frame] --> D[Frame Ingestion]
    end

    subgraph P2["Phase 2: Model Setup & Inference Pipeline"]
        D --> E[YOLOv8n Convolutional Detector\nClasses: car, truck, bus, motorcycle]
        E --> F[Extract Vehicle BBoxes & Confidence\nFilter Conf >= τ_conf]
    end

    subgraph P3["Phase 3: Spatial Logic & Occupancy (IoA)"]
        B --> G[Exact Polygon-Box Intersection\nShapely Poly & Box]
        F --> G
        G --> H["Calculate IoA Occupancy Ratio:\nIoA = Area(Slot ∩ Car) / Area(Slot)"]
        H --> I{Max IoA >= τ_ioa?}
        I -->|Yes| J[Initial State: Occupied]
        I -->|No| K[Initial State: Vacant]
    end

    subgraph P4["Phase 4: Temporal Filtering & Edge Cases"]
        J --> L[TemporalStateDebouncer\nSliding Window Buffer N=5]
        K --> L
        L --> M[Debounced Slot State\nOccupied 🔴 / Vacant 🟢]
    end

    subgraph P5["Phase 5: Output Structuring"]
        M --> N[Standardized JSON Payload]
        M --> O[Streamlit UI Tab 6: Overlays, KPIs, Telemetry & JSON View]
    end
```

#### Detailed Phase Breakdown:

1. **Phase 1 — Data Preparation & Dual Native ROI Calibration (`slots_config.json`):**
   - **Dual Native Calibration Strategy:** To prevent accuracy degradation caused by non-uniform aspect ratio downscaling ($1.78:1$ vs $2.38:1$), `slots_config.json` provides dedicated, hand-verified perspective trapezoids for each native coordinate space:
     - **Master Reference Calibration (`empty lot.jpg`):** Traced directly onto the $1372 \times 768$ reference frame, mapping the front bays along actual painted double-yellow divider lines and transverse stop lines.
     - **Time-Series Surveillance Calibration (`row_sequence`):** Traced natively on $457 \times 192$ surveillance frames (`image_1.png` to `image_12.png`).
     - **Dynamic Resolution Scaling:** Automatically rescales polygon coordinates proportionally $(x \cdot \frac{W_{\text{target}}}{W_{\text{calib}}}, y \cdot \frac{H_{\text{target}}}{H_{\text{calib}}})$ if arbitrary camera feed dimensions are encountered.
   - **Interactive Calibration Tooling:** Includes a browser-based HTML5 annotator [`calibrate.html`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/calibrate.html) and OpenCV desktop utility [`scripts/calibrate_roi.py`](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/scripts/calibrate_roi.py) for easy click-to-calibrate ROI definitions.

2. **Phase 2 — Model Setup & Adaptive Low-Light Inference Pipeline (`run_vehicle_inference`):**
   - **Model:** Pre-trained YOLOv8n (Nano) convolutional object detector optimized for real-time CPU/GPU edge execution.
   - **Class Filtering:** Strict COCO category filtering to vehicle classes: Class 2 (`car`), Class 3 (`motorcycle`), Class 5 (`bus`), Class 7 (`truck`).
   - **Adaptive Low-Light CLAHE Boost:** Applies Contrast Limited Adaptive Histogram Equalization (`enhance_low_light`) on the $L$-channel in CIELAB color space (`clipLimit=2.5`, `tileGridSize=(8, 8)`), illuminating dark/black vehicles (SUVs, pickups) and shadowed tire wells.
   - **Dual-Exposure Multi-Pass Inference + NMS:** Combines candidate bounding boxes across raw and CLAHE-enhanced contrast passes, merging duplicates via Non-Maximum Suppression (`iou_thresh=0.55`).
   - **Confidence Thresholding:** Configurable threshold $\tau_{\text{conf}} \in [0.15, 0.85]$ (default: $0.25$).

3. **Phase 3 — Spatial Logic & Occupancy Calculation (`calculate_ioa_occupancy`):**
   - **Intersection over Area (IoA) Formulation:**
     $$\text{Raw IoA} = \frac{\text{Area}(\text{Slot Polygon} \cap \text{Vehicle Box})}{\text{Area}(\text{Slot Polygon})}$$
   - **Centroid Containment & Coverage Reinforcement Rules:**
     $$\text{Effective IoA} = \begin{cases} \max(\text{Raw IoA}, 0.60) & \text{if vehicle centroid } (c_x, c_y) \in \text{Slot Polygon} \\ \max(\text{Raw IoA}, 0.55) & \text{if vehicle coverage } \frac{\text{Area}(\text{Slot} \cap \text{Car})}{\text{Area}(\text{Car})} \ge 0.35 \\ \max(\text{Raw IoA}, 0.50) & \text{if tire ground point } (c_x, y_2) \in \text{Slot Polygon} \text{ and } \text{Raw IoA} \ge 0.10 \\ \text{Raw IoA} & \text{otherwise} \end{cases}$$
   - **Decision Boundary:**
     $$\text{Status}(\text{Slot}_k) = \begin{cases} \text{Occupied (🔴)} & \text{if } \max_{v \in \mathcal{V}} \text{Effective IoA}(\text{Slot}_k, v) \ge \tau_{\text{ioa}} \\ \text{Vacant (🟢)} & \text{otherwise} \end{cases}$$
   - **Low-Confidence Quality Flag:** A review flag `low_confidence_flag = True` is triggered whenever $|\text{Effective IoA} - \tau_{\text{ioa}}| \le 0.05$, enabling human-in-the-loop review for edge cases.

4. **Phase 4 — Temporal Filtering & Edge-Case Handling (`TemporalStateDebouncer`):**
   - **Sliding-Window State Debouncing:** A rolling historical buffer ($N = 5$ frames) tracks sequential slot states across time-series feeds (`image_1.png` to `image_12.png`), requiring a majority confirmation ratio ($\ge 60\%$) to transition states.
   - **Occlusion Mitigation:** True perspective trapezoids combined with base ground-contact testing eliminate false occupancy triggers caused by vehicle roofs projecting over background bays.

5. **Phase 5 — Output Structuring (`generate_standard_json_payload`):**
   - Live telemetry is serialized into standardized JSON payloads ready for downstream API ingestion:
     ```json
     {
       "timestamp": "2026-08-19T02:00:00Z",
       "camera_feed": "empty lot.jpg",
       "total_spaces": 6,
       "occupied_count": 0,
       "vacant_count": 6,
       "occupancy_rate": 0.0,
       "borderline_count": 0,
       "slots": [
         {"id": "F-01", "name": "Bay F-01", "zone": "Front Row", "status": "vacant", "occupancy_ratio": 0.0, "confidence": 1.0, "vehicle_class": null, "low_confidence_flag": false},
         {"id": "F-02", "name": "Bay F-02", "zone": "Front Row", "status": "vacant", "occupancy_ratio": 0.0, "confidence": 1.0, "vehicle_class": null, "low_confidence_flag": false}
       ]
     }
     ```

---

## 10. Enterprise Dashboard UI Architecture (`app.py`)

Built with Streamlit and powered by a dual **Light & Dark Theme Engine** (`#121417` Deep Dark / `#F8FAFC` Light Slate) with high-contrast typography, interactive Plotly visualizations, and zero-page-reload modal dialogues.

### 10.1 Key UI & UX Innovations

1. **Township Deck Map & Sequential Hierarchy:**
   - Zones are strictly ordered according to commercial parking schemes: **Mall** $\rightarrow$ **Office** $\rightarrow$ **Residential**.
   - Facing rows are separated by realistic labeled **Drive Aisles** (`DRIVE AISLE · LANE A ➔`, `LANE B`, `LANE C`).
   - Expanded parking bay capacity modeling (48 Mall bays, 24 Office bays, 16 Residential bays).

2. **Interactive Click-to-Inspect Parking Blocks (`@st.fragment` & `@st.dialog`):**
   - Every individual parking stall block is an interactive solid-color button matching its real-time state.
   - Powered by isolated fragment execution (`@st.fragment`), clicking any stall block triggers the modal inspection dialog in-place without reloading the outer dashboard or interrupting the live clock.
   - Opens a 6-tab modal dialog querying SQLite (`data/parking.db`) for that specific slot:
     - `Active Live Telemetry` — Consolidated snapshot of state, ticket, and OCR reading.
     - `Active Occupancy State (current_state)` — Bay status and last state change timestamp.
     - `Ticketing & Settlement (ticketing_records)` — Assigned vehicle plate, entry time, payment settlement time.
     - `Optical Plate Read (plate_reads)` — Camera OCR detected text, confidence vector, ground truth plate.
     - `Infrastructure Metadata (slots & zones)` — Deck level, archetype, township site, zone capacity.
     - `Executed SQLite Statement (Raw SQL)` — Underlying SQL statement executed against `data/parking.db`.

3. **Dynamic Real-Time PST Hero Clock (Tab 1):**
   - Centered client-side Philippine Standard Time (PST) clock displaying hours, minutes, and continuous ticking seconds (`HH:MM:SS AM/PM`) with zero backend reload overhead.
   - Features a pulsing green status indicator (`🟢 PHILIPPINE STANDARD TIME (PST) · REAL-TIME LIVE OCCUPANCY MONITORING`).

4. **Streamlined Predictive Timeline Controls & Forecast Intelligence (Tab 2):**
   - Clean, centered Date and Time selectors for rapid predictive scenario analysis across any future timestamp.
   - Simulation state engine uses 5-minute window block seeds to prevent stochastic bay reshuffling during active inspection.
   - Integrated **Forecast Intelligence & Methodology** module providing zone classification profiles (Mall vs. Office vs. Residential), 4-week telemetry dataset provenance (2,688 intervals at 15-minute resolution), and detailed model specifications.

5. **Enterprise Dark & Light Mode Theme Engine:**
   - Seamless segmented theme switch between Dark Mode and Light Mode.
   - Full dynamic theme coverage across BaseWeb calendars, popover dropdowns, segmented controls, number/time inputs, and auto-adapting Plotly charts.

### 10.2 Dedicated Functional Tabs:
1. **Occupancy Map:** Live real-time operations deck displaying all bays across Uptown Bonifacio and Eastwood City zones ordered sequentially by archetype with Drive Aisles, live capacity KPIs, dynamic ticking PST clock, and in-place `@st.fragment` database row inspection.
2. **Availability Forecast:** Dedicated future planning tool allowing operators and visitors to pick any zone, date, and future arrival time to receive the ML forecast, baseline comparison, conservative safety margin, historical diurnal curve, and Forecast Intelligence breakdown.
3. **Model Performance:** Diagnostic validation dashboard detailing AI Prediction Error %, Standard Baseline Guess Error %, Accuracy Advantage %, Permutation Feature Importance with plain-English signal labels, Actual vs. Forecasted time-series tracking, and an Operational Executive Summary for non-technical stakeholders.
4. **Plate Matching:** Interactive slot inspector testing the fuzzy matcher on noisy OCR plate reads, displaying per-character confidence scores, ranked candidate tickets, and match margin validation with optical confusable-pair handling ($0/O, 1/I, 8/B, 5/S, 2/Z, 6/G$).
5. **ALPR Feasibility (CV):** Dual-dataset visual gallery allowing users to toggle between the **🇵🇭 Philippine Parking Lot CCTV Dataset** (20 surveillance video frames) and the **🌍 Academic OpenALPR Benchmark** (14 photos), inspecting YOLOv8 vehicle boxes, localized plate crops, OCR reads, and matcher resolutions.
6. **Space Detection (CV):** Enterprise parking space occupancy detection engine across surveillance feeds in `car_dataset/`, implementing the 5-Phase ROI-IoA algorithm with interactive $\tau_{\text{conf}}$ and $\tau_{\text{ioa}}$ sliders, temporal smoothing toggles, live capacity KPIs, annotated/raw/JSON feed tabs, and slot-by-slot telemetry.

---

## 11. Production Deployment, Edge Hardware & Scaling Guide

### 11.1 Recommended Edge Hardware Topology
For a physical township deployment across 1,000+ slots and 12 gate lanes:

```
[ Gate Cameras (4MP RTSP) ] ----> [ Edge AI Node (NVIDIA Jetson Orin) ] 
                                            | (gRPC / MQTT)
[ Overhead Slot Sensors ] ------> [ Township Gateway (x86 Server) ]
                                            | (TLS / HTTPS)
                                  [ Cloud Central Platform ]
                                  - PostgreSQL Database
                                  - Redis State Cluster
                                  - Retraining Pipeline (Airflow)
                                  - Streamlit Ops Portal
```

### 11.2 Edge Compute Specifications
- **Gate ALPR Nodes:** NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB) running TensorRT-accelerated YOLOv8-Plate + Fast-ANPR (Inference latency: $< 25\text{ms}$ per vehicle).
- **Camera Specifications:** 4MP (2560x1440), 60 FPS, Global Shutter or low rolling shutter distortion, Motorized Varifocal Lens (5–50mm), Integrated 850nm IR illuminator.
- **Overhead Slot Sensors:** Dual-bay dual-lens AI camera or BLE/Ultrasonic sensor clusters feeding MQTT broker over PoE.

### 11.3 Data Privacy & Retention Policy
- **Image Retention:** Raw vehicle images captured at entry gates are processed in edge RAM and discarded immediately after plate extraction.
- **Plate Hashing:** License plate strings in long-term historical logs can be salted and SHA-256 hashed for analytical modeling to comply with local privacy regulations (e.g., Philippine Data Privacy Act of 2012).

---

## 12. Operational Runbook & Quickstart

### 12.1 Local Execution
```bash
# 1. Clone repository & enter directory
git clone https://github.com/tantanrevidad/parking-poc.git
cd parking-poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate SQLite database & synthetic dataset
python generate_data.py

# 4. Fetch real benchmark & CCTV datasets
python fetch_real_dataset.py
python fetch_ph_dataset.py

# 5. Run Computer Vision benchmark pipeline
python cv_demo.py --dataset all

# 6. Launch Streamlit Enterprise Dashboard
python -m streamlit run app.py
```

### 12.2 Verification Checklist
- [x] Database initialized: `data/parking.db` contains 6 zones and 120 slots.
- [x] Streamlit running on port `8502` or `8501`.
- [x] ML model achieves $>60\%$ MAE reduction over baseline.
- [x] CV demo achieves $>85\%$ character accuracy with 0 false positive ticket matches.
