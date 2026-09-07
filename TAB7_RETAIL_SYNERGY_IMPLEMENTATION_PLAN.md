# Tab 7 Revision — Retail Spend Synergy Implementation Plan

**Scope of this revision:** remove the dynamic-pricing simulator entirely. Rebuild
Tab 7 around three data-backed insight modules: (A) Dwell → Spend Elasticity
(existing, retained), (B) Parking Friction & Saturation Model (new), and
(C) Repeat-Visitor Recognition Engine (new). Revenue Leakage (existing) is kept
but moved to the end of the tab, since it's an operations module rather than a
retail-synergy one.

**Design principle carried over from the rest of the app:** every number shown
must trace to a labeled provenance tier. This plan introduces one new tier:

| Tier | Badge | Meaning |
|---|---|---|
| **E** (new) | `MODELED` | An explicit, clearly-labeled assumption used to translate a real research finding into a formula, where the source study didn't publish an exact coefficient. Distinguished from Tier D so we never overstate how precise a borrowed number is. |

---

## Phase 0 — Prerequisite: fix the synthetic data generator (blocking)

**This must happen before Section C can show anything meaningful.**

`generate_data.py::random_plate()` currently draws a fresh random 3-letter/
4-digit plate **for every ticket, independently**. With ~17.6M possible
combinations, the probability of the same plate appearing twice in 28 days of
synthetic data is effectively zero. If we build the Repeat-Visitor engine
against the current generator, the Repeat Visitor Rate KPI will read ~0%
forever — which would quietly undermine the entire pitch.

**Fix:** introduce a finite per-site "vehicle pool" with a skewed draw
probability, so a minority of plates account for a disproportionate share of
visits (mirroring real repeat-customer distributions, e.g. Pareto-style
80/20 skew).

```python
# generate_data.py — replace per-ticket random_plate() calls with a pool draw

def build_vehicle_pool(site_name, pool_size=1400, seed=None):
    """
    Generates a fixed pool of plates for a site, with a Zipf-distributed
    draw weight so a minority of vehicles return disproportionately often.
    pool_size is calibrated so that, given the site's daily ticket volume
    over the 28-day window, the resulting repeat-visit rate lands in a
    realistic 20-35% range (tunable via ZIPF_EXPONENT).
    """
    rng = random.Random(seed)
    plates = [random_plate() for _ in range(pool_size)]
    ranks = np.arange(1, pool_size + 1)
    weights = 1.0 / np.power(ranks, ZIPF_EXPONENT)  # e.g. ZIPF_EXPONENT = 1.1
    weights = weights / weights.sum()
    rng.shuffle(plates)  # decorrelate rank from plate string
    return plates, weights

# When generating a ticket:
plate = np.random.choice(vehicle_pool, p=vehicle_pool_weights)
```

Tag every generated ticket's provenance as **Tier B (Derived / synthetic,
statistically calibrated)** same as today — nothing about the provenance
framework changes, we're just making the synthetic generator produce a
realistic repeat-visit distribution instead of an accidentally-unrealistic one.

**Acceptance check for Phase 0:** after regeneration, `SELECT plate,
COUNT(*) FROM ticketing_records GROUP BY plate ORDER BY 2 DESC` should show a
long tail with a meaningful head (e.g., top 10% of plates responsible for
~30-40% of visits). If the distribution is flat, the pool/Zipf parameters need
adjusting before moving to Section C.

---

## Section A — Dwell → Spend Elasticity (retain, minor cleanup only)

**No formula changes.** `dwell_spend_elasticity = 1.3` in
`revenue_config.py` already matches the published Path Intelligence finding
(1% dwell increase → ~1.3% sales increase) and should stay exactly as is.

**Minor changes:**
1. Update the `source` string in `INDUSTRY_BENCHMARKS["dwell_spend_elasticity"]`
   to cite Path Intelligence explicitly by name (currently attributes it more
   generally to ICSC) so the provenance tooltip is precise.
2. Rename the UI heading from "Section D: Retail Intelligence & Township
   Economic Synergy" to **"Section B: Dwell → Spend Elasticity"** as part of
   the tab reorder (see Final Tab Layout below).
3. No changes to `compute_dwell_distribution()` or the spend-projection logic.

---

## Section B → renumbered Section C — Parking Friction & Saturation Model (new)

### Research basis
A 2014 ScienceDirect study, *"Do Parking Fees Affect Retail Sales? Evidence
from Starbucks,"* found that the effect of a parking fee on nearby retail
sales flips sign depending on whether parking is saturated: when parking is
saturated, higher fees generate turnover that increases customer flow, which
can raise total sales even though individual shopping time drops. When
parking is not saturated, higher charges simply shorten shopping time without
increasing available parking, which reduces individual purchases and total
retail sales.

The exact numeric coefficient linking "fee level" to "shopping-time
reduction" isn't published in a form we can drop into a formula — so this
module is explicitly built as a **Tier E (Modeled)** diagnostic, not a
Tier A/B fact. That's a feature, not a limitation: it should be presented to
the exec as "here's where your current static rate card is working against
you, and where it isn't" — a risk map, not a forecast.

### Formula
For each zone × hour bucket, using existing `occupancy_history`:

```
occ_rate = occupied_count / capacity          # already computed today

if occ_rate >= SATURATION_THRESHOLD (default 0.85):
    regime = "Saturated"
    # Fee is not suppressing visits — turnover is the binding constraint,
    # not price. No revenue-at-risk flag.
    spend_at_risk = 0

else:
    regime = "Unsaturated"
    # Modeled (Tier E) friction coefficient: fraction of the current fee's
    # "bite" attributable to discouraging a marginal visit, applied through
    # the existing (Tier C) dwell-spend elasticity from Section A.
    friction_factor = FRICTION_COEFFICIENT  # e.g. 0.15, exposed as a labeled
                                             # sensitivity slider in the UI —
                                             # never presented as a fixed fact
    implied_dwell_suppression_pct = friction_factor * (current_fee / reference_fee)
    spend_at_risk = implied_dwell_suppression_pct * dwell_spend_elasticity \
                    * avg_spend_per_visit * visits_in_bucket
```

`reference_fee` is the zone's own first-3-hour flat rate (Tier A), so the
ratio is self-normalized per township and needs no cross-site calibration.

### New function — `revenue_engine.py`

```python
def compute_friction_saturation_model(
    site_name: str,
    date_str: str,
    saturation_threshold: float = 0.85,
    friction_coefficient: float = 0.15,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Classifies each zone x hour bucket as Saturated / Unsaturated and
    estimates spend-at-risk in Unsaturated buckets, per the Starbucks
    parking-fee study's regime-dependent effect.

    Data Provenance:
        - Occupancy: Tier B (occupancy_history)
        - Fee ratio: Tier A (PARKING_RATES)
        - Friction coefficient: Tier E (MODELED — user-adjustable, not a
          published constant)
        - Dwell-spend conversion: Tier C (Path Intelligence elasticity,
          reused from Section A)
    """
```

Returns a per-zone-hour DataFrame plus a rollup: total estimated spend-at-risk
for the day, and a ranked list of the worst-offending zone/hour combinations
(the ones an operator could plausibly act on, e.g. "Residential visitor bays,
weekday mornings — 40% unsaturated, ₱X/day spend at risk").

### New config — `revenue_config.py`

```python
FRICTION_MODEL_PARAMS: Dict[str, Any] = {
    "saturation_threshold": {
        "value": 0.85,
        "source": "ScienceDirect: 'Do Parking Fees Affect Retail Sales? "
                   "Evidence from Starbucks' (2014); consistent with Shoup's "
                   "85% occupancy target used elsewhere in this dashboard.",
        "provenance": "Tier D",
    },
    "friction_coefficient": {
        "value": 0.15,
        "default_range": [0.05, 0.30],
        "note": "Modeled sensitivity parameter, not a published figure. "
                "Exposed as an adjustable slider in the UI so the estimate "
                "is never presented as more precise than it is.",
        "provenance": "Tier E",
    },
}
```

### UI (Streamlit)
- Small-multiple heatmap identical in style to Section A's existing heatmap,
  but colored by regime (two-tone: Saturated / Unsaturated) instead of ₱.
- A single slider for `friction_coefficient`, labeled explicitly
  "Modeled assumption — adjust to test sensitivity" so nobody mistakes it
  for a measured constant.
- A ranked table: top 5 zone/hour buckets by spend-at-risk, each with a
  one-line explanation ("Unsaturated 62% of the time, 09:00–11:00 weekdays").

---

## Section C → renumbered Section D — Repeat-Visitor Recognition Engine (new)

### Research basis
Two aggregated industry findings anchor this module:
1. A widely-cited Bain & Company finding: increasing customer retention by a small amount can lift profits substantially, with estimates in the 25% to 95% range for a 5-point retention improvement.
2. Retail-loyalty research aggregated by multiple retention-analytics firms
   consistently finds returning customers spend roughly two-thirds more per
   visit than first-time customers — this is a compiled industry figure
   (Tier D), not a single peer-reviewed source, and should be labeled as such.

The point of this module: **the property already has the data to measure
this, for free**, because `matcher.py`'s plate-matching logic exists for a
different purpose (resolving noisy OCR reads at exit). We are not building
new capture infrastructure — we're running a new query against
`ticketing_records`, which already has a clean `plate` field once a ticket is
settled.

### Privacy note (include in the plan, not an afterthought)
`SYSTEM_DOCUMENTATION.md` already flags that plate strings in long-term
historical logs should be salted and hashed to comply with the Philippine
Data Privacy Act of 2012. This module is exactly the case that requirement
was written for — it must **not** display or store raw plate strings. All
grouping/analytics run on a salted SHA-256 hash of the plate, computed at
query time (or at ingestion, if this becomes a production system). No plate
string should ever render in the Tab 7 UI for this feature.

```python
import hashlib

def hashed_plate(plate: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{plate}".encode()).hexdigest()[:16]
```

### Formula / segmentation logic
```
lookback_days = 28   # full available synthetic window; make this a
                      # UI-selectable parameter (7 / 14 / 28) for flexibility

visits_per_plate = COUNT(ticket_id) GROUP BY hashed_plate
                   WHERE entry_time >= today - lookback_days

segment:
    "New"        if visits_per_plate == 1
    "Returning"  if 2 <= visits_per_plate <= 3
    "Loyal"      if visits_per_plate >= 4

repeat_visitor_rate = tickets_from(Returning + Loyal) / total_tickets

# Incremental spend attributable to the returning/loyal cohort, using the
# same avg_spend_per_visit baseline already used in Section A/B, plus the
# industry repeat-spend premium (Tier D):
incremental_spend = tickets_from(Returning + Loyal) \
                     * avg_spend_per_visit * (repeat_spend_premium)  # e.g. 0.67
```

### New module — `loyalty_engine.py`

```python
"""
loyalty_engine.py
=================
Repeat-visitor segmentation and estimated loyalty spend value.
Operates on hashed plates only (PH Data Privacy Act 2012 compliant) —
never surfaces raw plate strings.
"""

def compute_repeat_visitor_segments(
    site_name: Optional[str] = None,
    lookback_days: int = 28,
    salt: str = DEFAULT_SALT,       # from st.secrets / env, not hardcoded
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns:
      - repeat_visitor_rate: float
      - segment_counts: {"New": n, "Returning": n, "Loyal": n}
      - segment_ticket_share: {...}          # % of total visits per segment
      - estimated_incremental_spend_php: float
      - top_loyal_zone_types: [...]          # which zone types skew most repeat
      - provenance: str
    """
```

### New config — `revenue_config.py`

```python
LOYALTY_BENCHMARKS: Dict[str, Any] = {
    "retention_profit_uplift_pct": {
        "low": 25.0,
        "high": 95.0,
        "basis": "5-percentage-point retention increase",
        "source": "Bain & Company customer retention research (widely cited "
                   "industry figure)",
        "provenance": "Tier D",
    },
    "repeat_customer_spend_premium_pct": {
        "value": 67.0,
        "source": "Aggregated retail loyalty / retention-analytics industry "
                   "research (compiled figure, not a single primary study)",
        "provenance": "Tier D",
    },
}
```

### UI (Streamlit)
- Three KPI cards: Repeat Visitor Rate, Loyal-segment share of total visits,
  Estimated incremental spend (₱) attributable to repeat visitors — labeled
  `MODELED` combining Tier B (real segment counts) with Tier D (spend premium).
- A stacked bar: New / Returning / Loyal share of visits, by zone type — this
  is likely to be the most interesting chart for an exec, since it'll probably
  show malls skew more repeat than office/residential visitor bays, which is
  a genuinely new, non-obvious insight the property doesn't currently have.
- A footnote disclosing the hashing methodology, for anyone from Legal/Data
  Privacy who reviews the deck.

---

## File-by-file change summary

| File | Change |
|---|---|
| `generate_data.py` | Add `build_vehicle_pool()`; replace per-ticket `random_plate()` calls with weighted pool draws (Phase 0, blocking) |
| `revenue_config.py` | Add `PROVENANCE_TIERS["E"]`; add `FRICTION_MODEL_PARAMS`; add `LOYALTY_BENCHMARKS`; update `dwell_spend_elasticity` source string |
| `revenue_engine.py` | Add `compute_friction_saturation_model()`; **remove** `simulate_dynamic_pricing()` and its `revenue_config` references |
| `loyalty_engine.py` | **New file.** `compute_repeat_visitor_segments()`, `hashed_plate()` |
| `app.py` | Remove old "Section B: Dynamic Pricing & Yield Simulator" block (incl. `render_dynamic_pricing_fragment`); reorder/relabel remaining sections per Final Tab Layout below; add two new rendered sections |
| `.streamlit/secrets.toml` (or env) | Add `PLATE_HASH_SALT` — do not hardcode the salt in source |

## Final Tab 7 layout

1. **Revenue Operations Deck** (unchanged — KPIs + zone×hour ₱ heatmap)
2. **Section B: Dwell → Spend Elasticity** (renamed from current Section D; logic unchanged)
3. **Section C: Parking Friction & Saturation Model** (new)
4. **Section D: Repeat-Visitor Recognition Engine** (new)
5. **Revenue Leakage & Overstay Recovery** (unchanged — moved to the end, it's operational rather than retail-synergy)

---

## Testing / validation checklist

- [ ] Phase 0: regenerate DB, confirm repeat-visit distribution is non-trivial (see acceptance check above)
- [ ] Section C: spot-check 2-3 manually computed zone/hour buckets against `compute_friction_saturation_model()` output
- [ ] Section C: confirm slider changes to `friction_coefficient` visibly move spend-at-risk numbers (sanity check that it's wired, not decorative)
- [ ] Section D: confirm no raw plate string ever appears in rendered HTML/DOM (inspect via browser devtools, not just visually)
- [ ] Section D: confirm hashed plates are stable across the session (same car → same hash) but salt is not committed to source control
- [ ] `streamlit run app.py` — full Tab 7 render with no errors, all provenance badges present
- [ ] Remove all dead imports/functions related to the old pricing simulator (`simulate_dynamic_pricing`, related Plotly figures, related session-state keys)

---

## Suggested build order

1. Phase 0 (generator fix) — everything else is blocked on this
2. Section A cleanup (fast, low-risk, do it first to get a quick win)
3. Section C (Friction model) — pure read-only analytics on existing tables, no new schema
4. Section D (Repeat-Visitor) — depends on Phase 0 data + new hashing utility
5. Tab reorder + removal of old pricing simulator code
6. Full-tab QA pass against the checklist above
