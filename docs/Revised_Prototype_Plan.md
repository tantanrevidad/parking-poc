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

**Adopted addition:** a fifth tab, **"How the System Senses Parking,"**
displaying Component 3's real annotated images and matching results —
matching the source plan's Step 4.3 intent, but as a tab in the existing
app rather than a section of a hand-built static page.

**Rejected: rebuilding as a single static HTML/React artifact.**
Considered and explicitly not adopted. What it would gain — zero-install,
opens directly in a browser — is real but smaller than it looks, since
`pip install -r requirements.txt && streamlit run app.py` is not a
meaningfully higher bar for a review audience than opening a static file.
What it would cost is larger and directly cuts against what you asked for
going into this build: baking predictions into a static JSON lookup means
giving up the live-trained-model story (real holdout metrics computed on
the spot, not a frozen snapshot), and hand-building a seat-map grid, tabs,
and charts in vanilla HTML/JS would be real, non-trivial engineering effort
to reach rough parity with what Streamlit already provides. Not worth the
trade.

---

## Explicit Boundaries of This Prototype (state plainly when presenting)

- Occupancy/ticketing/predictive data across all sites is **synthetically
  generated** — realistic in shape, not observed fact.
- The CV demo (Component 3) runs on **real photographs**, but generic
  public ones, not actual Megaworld structures or plates — and is
  **deliberately not wired into** the live app's data, since there's no
  real camera feed to connect to a fictional township.
- The ML model is **real and trained live** each time the app starts (or
  on-demand via "Regenerate synthetic dataset") — not a frozen export —
  which is a deliberate choice to keep the "watch it work" story intact.
- This remains a proof-of-concept for the *idea*, not production
  infrastructure. The full implementation plan's phased roadmap still
  governs the real path to deployment.

---

## Revised Build Order

1. **Enrich Component 1** — multi-site, zone-type curves in
   `generate_data.py`. Everything downstream should keep working unchanged
   since it already reads zones generically.
2. **Re-verify Component 2** — confirm the model's improvement-over-baseline
   number with the richer data; update the Model Insights tab copy if the
   number moves.
3. **Build Component 3** — standalone CV demo folder (YOLOv8n detection +
   Tesseract OCR w/ plate localization + reused `matcher.py`), producing
   annotated images and a results JSON.
4. **Add the "How the System Senses Parking" tab** to the existing app,
   embedding Component 3's output.

Steps 1–2 and Step 3 can happen in either order or in parallel — Component
3 has no dependency on Components 1/2. Step 4 goes last since it needs real
output from Step 3 to embed, not a placeholder.