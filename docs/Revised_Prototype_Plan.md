# Smart Parking Prototype — Revised Build Plan
### Merged Version: Existing Streamlit POC + Adopted Improvements

This supersedes both prior plans. It keeps what's already built and working
(the Streamlit app), adopts two genuinely valuable additions from the
`Demo_Prototype_Plan.md` review, and explicitly rejects the parts that
would trade away things we already have working for no clear net gain.

**Status of all components:**
- ✅ **Component 1 (Synthetic Data Generator):** Multi-township (Uptown, Eastwood), 3 zone types (Office, Mall, Residential) with diurnal curves.
- ✅ **Component 2 (ML Forecasting Model):** Gradient-boosted model training live with holdout evaluation and feature importance.
- ✅ **Component 3 (CV Feasibility Pipeline):** YOLOv8n + plate localization + OCR + zero false-positive matcher on dual datasets (Academic OpenALPR & Philippine CCTV surveillance).
- ✅ **Component 4 (Enterprise Web App):** 5-tab dark theme dashboard with interactive simulation clock, trip planner, model diagnostics, matcher inspector, and CV gallery.

---

## Component 1 — Synthetic Data Generator ✅ Done

**Current state:** `generate_data.py` produces one site, two generic zones,
28 days of 15-minute occupancy data with a single daily curve shape, plus a
small holidays/events calendar. This works and is what the ML model and
live seat map are already trained/driven on.

**Adopted improvement (from Demo_Prototype_Plan.md, Step 1.1–1.3):**
Expand to **two mock townships** (Uptown, Eastwood) with **zone types**
(`mall`, `office`, `residential`), each with a genuinely different daily
occupancy curve:
- Office: fills ~8–9am, dips at lunch, drains after 6pm, quiet on weekends.
- Mall: quiet mornings, builds through the afternoon, peaks in the evening,
  busier on weekends.
- Residential: high overnight, dips during work hours, flat across weekdays/weekends.

**Why this is worth doing:** right now every zone follows the same curve, so
the ML model's "beats the naive baseline" result is a weaker claim — there
isn't much structure to learn beyond time-of-day. Real heterogeneity across
zone types gives the model an actual pattern to earn credit for finding
(e.g., correctly learning that an office zone's Saturday looks nothing like
its Tuesday), which makes the Model Insights tab's story more convincing.

**Not adopted:** a `sites_config.json` as a separate file — the existing
SQLite schema already serves as the single source of truth shared between
the generator, the app, and the simulator; adding a parallel JSON config
would just be two sources of truth to keep in sync; that keeps the current
code architecture simpler than the source plan's.

**Effort:** small — this is an edit to `generate_data.py`'s curve function
and zone table, not a rewrite. Everything downstream (matcher, state
machine, predictor, app) already operates generically on "whatever zones
are in the `zones` table," so it should require no changes elsewhere.

**Output:** updated `data/parking.db` with 2 sites × multiple zone
types/levels, same schema as today.

---

## Component 2 — ML Forecasting Model ✅ Done (minor note)

**Current state:** `predictor.py` trains a real
`HistGradientBoostingRegressor` on a time-based holdout, reports MAE against
a naive (site, zone, hour, day-of-week) baseline, and exposes permutation
feature importance — all live in the Streamlit "ML Model Insights" tab.

This already satisfies everything Demo_Prototype_Plan.md's Component 2
describes (Steps 2.1–2.3), including the "baseline comparison is the most
important number" framing.

**Not adopted:** exporting predictions to a static `predictions.json`
lookup grid. That step exists in the source plan specifically to support a
backend-less static HTML artifact (see Component 4 below). Since we're
keeping Streamlit (which runs the trained model live), there's no reason to
freeze predictions into a lookup table — doing so would mean losing the
"watch it train, see the real holdout metrics right now" story, which is
more compelling for a review than a pre-baked answer.

**One addition worth making once Component 1 is enriched:** re-verify the
model's improvement-over-baseline percentage with the richer, more
heterogeneous zone data — it should go up, not down, and that's a good
number to have ready for the presentation.

---

## Component 3 — Computer Vision Feasibility Demo ✅ Done

**This is the main adopted addition.** Everything in the current app —
occupancy, plate reads, OCR confidence — is a generated number; nothing
touches a real pixel. A short standalone demo that runs real detection and
real OCR against real (generic, public) photos answers a different, harder
question than the rest of the POC does: *can this be done at all, cheaply?*

Kept from the source plan almost as written, with one substitution:

| Step | Plan | Decision |
|---|---|---|
| 3.1 Vehicle detector | Haar-cascade | **Swapped for YOLOv8n** (`ultralytics`, one pip install, CPU-only, filtered to `car`/`truck` classes). Public Haar-cascade car classifiers are unreliable enough at odd angles that they risk undermining the demo instead of proving feasibility — still a one-line, no-training-required setup, but produces convincing boxes. |
| 3.1 Image sources | Public GitHub repos (parking-lot dataset, plate-recognition tutorial dataset) | **Kept as-is** — no scraping, no licensing concerns, appropriate for a feasibility demo. |
| 3.2 Occupancy pass | Bounding boxes + count, honest caveat about detector limits at odd angles | **Kept as-is** — the "accepted blind-spot rate, no 100% coverage" framing directly mirrors Section 6.3 of the original implementation plan and is worth keeping in the write-up. |
| 3.3 Plate reading | Tesseract OCR, with plate-region localization (edge detection + rectangular contour search) *before* OCR | **Kept as-is, and kept mandatory** — this was correctly flagged as the difference between a convincing demo and a noisy one; whole-photo OCR is not an acceptable shortcut. |
| 3.4 Matching demo | Confidence-weighted matching against a small candidate pool, showing a confident match, a below-threshold non-match, and a near-tie | **Kept as-is — and reuse the existing `matcher.py`** rather than reimplementing matching logic from scratch. The confusable-character handling (0/O, 1/I, 8/B, 5/S, 2/Z) and the near-tie rejection are already implemented and unit-tested in the current codebase; Component 3 should call that module against real OCR output instead of duplicating the algorithm. |
| 3.5 Packaging | Annotated images + results JSON | **Kept as-is.** |

**Explicitly standalone, not wired into the live app:** there's no real
camera feed to connect to a fictional township, so this stays a separate
artifact — a folder of annotated images plus a results summary, embedded
into the Streamlit app as a static gallery/tab rather than driving any live
data. This avoids pretending the fake sites have real cameras.

**Output:** `cv-demo/` folder — annotated detection images (before/after),
`matching_results.json` (OCR read, matched plate or "unresolved",
confidence, reasoning per sample image), clearly labeled throughout as a
feasibility proof on generic public images, not Megaworld data.

---

## Component 4 — Interactive Web App ✅ Done

**Current state:** the Streamlit app already delivers the seat-map view,
trip-planner view (baseline vs. trained model, plain-language label), and
ML insights view — i.e., all of Demo_Prototype_Plan.md's Steps 4.1, 4.2,
and the metrics-transparency part of 4.2.

- ✅ **Component 4 (Enterprise Web App):** 6-tab dark theme dashboard with interactive simulation clock, trip planner, model diagnostics, matcher inspector, and dual CV galleries.
- ✅ **Component 5 (Multi-Angle Parking Space Vacancy Detection):** Real-time YOLOv8 vehicle detection + spatial bay IoU classification across `car_dataset/` camera perspectives.

---

## Component 5 — Multi-Angle Space Vacancy Detection ✅ Done

**Overview:** Adds computer vision parking space detection across multi-angle surveillance feeds in `car_dataset/`:
- **Multi-Angle Camera Feeds:** Evaluates 3 distinct CCTV perspectives (North Upper Deck, East Central Lot, South Perimeter) and 12 sequential slot row feeds.
- **Vehicle Object Detection:** Uses YOLOv8n to locate all cars, SUVs, trucks, and buses.
- **Spatial Overlap & IoU Classification:** Calculates intersection between designated parking bays and vehicle bounding boxes to classify bays as **VACANT (🟢)** or **OCCUPIED (🔴)**.
- **Interactive UI Tab:** Tab 6 in `app.py` allows interactive threshold tuning, visual feed toggling, live KPI tracking, and bay-by-bay telemetry.

---

## Explicit Boundaries of This Prototype (state plainly when presenting)

- Occupancy/ticketing/predictive data across all sites is **synthetically
  generated** — realistic in shape, not observed fact.
- The CV demo (Component 3 & Component 5) runs on **real photographs & surveillance feeds**, but generic
  public ones, not actual Megaworld structures or plates — and is
  **deliberately not wired into** the live app's simulated township data, since there's no
  real camera feed to connect to a fictional township.
- The ML model is **real and trained live** each time the app starts (or
  on-demand via "Regenerate synthetic dataset") — not a frozen export —
  which is a deliberate choice to keep the "watch it work" story intact.
- This remains a proof-of-concept for the *idea*, not production
  infrastructure. The full implementation plan's phased roadmap still
  governs the real path to deployment.