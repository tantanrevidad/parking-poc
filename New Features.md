# Revenue Intelligence Tab — Implementation Plan

> **Goal**: Add a Tab 7 "Revenue Intelligence" to the parking POC that demonstrates AI-driven revenue generation aligned with Megaworld's township business model. **Every figure shown must trace back to a verifiable data source — no invented numbers.**

---

## Data Provenance Framework

> [!IMPORTANT]
> This is the foundation that makes the feature credible. Every number in the Revenue Intelligence tab will carry a **provenance badge** indicating where it comes from.

We use **4 tiers of data sourcing**, each clearly labeled in the UI:

| Tier | Label | What It Means | Example |
|------|-------|---------------|---------|
| **A** | `ACTUAL RATE` | Real Megaworld parking rates, researched and verified | ₱50/first 3hrs at Uptown Mall |
| **B** | `DERIVED` | Computed from existing POC data using real-world formulas | Revenue = occupancy × rate × hours |
| **C** | `PH BENCHMARK` | Philippine industry data with citation | ₱1,000–₱3,000 avg mall spend/visit |
| **D** | `INDUSTRY` | International research with academic/professional citations | "Dynamic pricing yields 20–40% uplift" — SFpark/HAH Parking |

Every KPI card, chart, and metric in the tab will show a small provenance badge so reviewers instantly know: **"this number is real"** vs. **"this number is computed from real inputs."**

---

## Real Data Sources — What We Actually Have

### Tier A: Actual Megaworld Parking Rate Structures (Researched)

These are the **real, currently enforced** parking rates at each township:

#### Uptown Bonifacio (Uptown Mall)
| Time Block | First 3 Hours | 4th–7th Hour | 7th Hour+ |
|-----------|---------------|-------------|-----------|
| 6:00 AM – 12:00 NN | ₱50 flat | ₱15/hr | ₱100/hr |
| 12:01 PM – 5:59 AM | ₱50 flat | ₱15/hr | ₱30/hr |
| Overnight surcharge | +₱200 if enter before 12MN, leave after 12NN next day |

#### Eastwood City (Eastwood Mall)
| Day | First 3 Hours | Succeeding | Overnight |
|-----|--------------|-----------|-----------|
| Mon–Fri | ₱60 flat | ₱20/hr | +₱150 |
| Sat–Sun–Holidays | ₱60 flat rate (all day) | — | +₱150 |

#### McKinley Hill (Venice Grand Canal Mall)
| Period | First 3 Hours | Succeeding | Overnight |
|--------|--------------|-----------|-----------|
| Standard (Mon–Sun) | ₱50 flat | ₱20/hr | +₱150 |
| Grace Period | 15 mins drop-off/pick-up | — | — |

> **Source**: Megaworld Lifestyle Malls official parking signage, MoneyMax.ph rate compilations, social media advisories (2024–2025).

### Tier B: Existing POC Data We Can Compute Revenue From

Your `generate_data.py` already produces:

| Data | What It Contains | Revenue Relevance |
|------|-----------------|-------------------|
| `occupancy_history` | 24,192 rows: 15-min occupancy rate × 9 zones × 28 days | **Occupied bay-hours per zone per day** — multiply by rate = revenue |
| `ticketing_records` | `entry_time`, `payment_settled_at`, `slot_id` per ticket | **Actual dwell time per vehicle** — determines which rate tier applies |
| `current_state` | Real-time slot status with `updated_at` timestamps | **Overstay detection** — time in `occupied_unpaid` state |
| `zones` | `zone_type` (mall/office/residential), `capacity` per zone | **Rate assignment** — different rates per zone type |
| `events` | Megaworld sale/concert events with `impact` level | **Surge pricing triggers** |
| `holidays` | Philippine statutory holidays | **Holiday rate modifiers** |

**Key insight**: By applying the real Tier A rate structures to the Tier B occupancy/ticketing data, **every revenue figure is mathematically derived from real rates × simulated-but-realistic occupancy**. The occupancy shapes are modeled on real diurnal curves, so the revenue outputs are realistic estimates, not random numbers.

### Tier C: Philippine Industry Benchmarks

| Metric | Value | Source |
|--------|-------|--------|
| Avg Filipino mall spend per visit | ₱1,000 – ₱3,000 | Industry surveys, Colliers PH |
| Megaworld daily foot traffic (2025) | 297,000 across all Lifestyle Malls | Megaworld 2025 Annual Report |
| Megaworld mall leasing revenue (2025) | ₱6.9 billion (+9% YoY) | Megaworld FY2025 Financials |
| Megaworld total leasing revenue (2025) | ₱22 billion (+11% YoY) | Megaworld FY2025 Financials |
| Dwell time → spend correlation | +1% dwell time ≈ +1.3% retail spend | Retail analytics industry data |

### Tier D: International Industry Research

| Metric | Value | Source |
|--------|-------|--------|
| Dynamic pricing revenue uplift | 20–40% avg; up to 68–144% in case studies | HAH Parking case study, SFpark pilot |
| Cruising time without guidance | 12–20 minutes average | Multiple urban mobility studies |
| Cruising time with smart parking | ~2 minutes | Smart parking operator benchmarks |
| Revenue leakage in manual facilities | 5–15% of theoretical max revenue | Industry white papers (Vert.ai, ParkLink) |
| SFpark cruising reduction | 30% reduction in search traffic | San Francisco MTA SFpark evaluation |

---

## Proposed Changes

### New Modules

---

#### [NEW] `revenue_config.py`

The single source of truth for all rate structures. **Every peso amount traces back to Tier A research.**

```python
PARKING_RATES = {
    "Uptown Bonifacio": {
        "mall": {
            "source": "Uptown Mall official parking signage (2024-2025)",
            "structure": "tiered",
            "first_3hr_flat": 50,      # ₱50 for first 3 hours
            "4th_to_7th_hr": 15,       # ₱15/hr for hours 4-7
            "7th_hr_plus_am": 100,     # ₱100/hr if entered 6AM-12NN
            "7th_hr_plus_pm": 30,      # ₱30/hr if entered 12:01PM-5:59AM
            "overnight_surcharge": 200, # ₱200 additional
        },
        "office": { ... },  # Same structure, may differ
        "residential": { ... },  # Flat monthly / daily rates
    },
    "Eastwood City": {
        "mall": {
            "source": "Eastwood Mall / MoneyMax.ph (2024-2025)",
            "structure": "tiered",
            "first_3hr_flat": 60,
            "succeeding_hr_weekday": 20,
            "weekend_holiday_flat": 60,  # Flat rate all day
            "overnight_surcharge": 150,
        },
        ...
    },
    "McKinley Hill": { ... },
}

INDUSTRY_BENCHMARKS = {
    "avg_mall_spend_per_visit_php": {"low": 1000, "mid": 2000, "high": 3000,
        "source": "Philippine retail industry surveys, Colliers PH"},
    "dwell_spend_elasticity": {"value": 1.3,
        "meaning": "1% increase in dwell time ≈ 1.3% increase in retail spend",
        "source": "Retail analytics industry research"},
    "dynamic_pricing_uplift_pct": {"conservative": 15, "moderate": 25, "aggressive": 40,
        "source": "HAH Parking case studies; SFpark municipal pilot"},
    "manual_leakage_rate_pct": {"low": 5, "mid": 10, "high": 15,
        "source": "Vert.ai, PreciseParkLink industry white papers"},
    "avg_cruising_minutes_without": 16,
    "avg_cruising_minutes_with": 2,
    "co2_per_liter_gasoline_kg": 2.31,
    "avg_fuel_consumption_idling_lph": 0.9,  # liters per hour while searching
}
```

---

#### [NEW] `revenue_engine.py`

Core computation module. Every function documents its formula and data provenance.

**Key functions:**

| Function | What It Computes | Data Sources |
|----------|-----------------|-------------|
| `compute_ticket_revenue(entry_time, exit_time, site, zone_type)` | Revenue for a single parking session using real tiered rate structure | Tier A rates × Tier B timestamps |
| `compute_zone_daily_revenue(zone_id, date)` | Sum of all ticket revenues in a zone for a day | Aggregates ticket-level revenue |
| `compute_rpbh(zone_id, date)` | Revenue Per Bay Per Hour = daily revenue ÷ (capacity × operating_hours) | Derived metric |
| `compute_revenue_heatmap(site_id, date_range)` | Zone × Hour matrix of revenue intensity | Occupancy history × rates |
| `simulate_dynamic_pricing(zone_id, date, strategy)` | Projects revenue under 3 pricing strategies | ML predictions × pricing functions |
| `detect_overstay_leakage(threshold_hours)` | Flags vehicles in `occupied_unpaid` > threshold, computes lost revenue | FSM state timestamps × rates |
| `compute_dwell_distribution(zone_id)` | Histogram of parking durations from ticketing data | `entry_time` to `payment_settled_at` |
| `estimate_visitor_retail_spend(dwell_hours)` | Estimated mall spend based on dwell time and PH benchmark | Tier C benchmarks × Tier B dwell |

**Revenue computation formula** (transparent and auditable):

```python
def compute_ticket_revenue(entry_time, exit_time, site_name, zone_type):
    """Compute parking fee using ACTUAL Megaworld rate structure.
    
    Data Provenance:
        - Rate structure: Tier A (real Megaworld parking rates)
        - Timestamps: Tier B (from ticketing_records table)
        - Formula: Tier B (deterministic calculation)
    """
    rates = PARKING_RATES[site_name][zone_type]
    dwell_hours = (exit_time - entry_time).total_seconds() / 3600
    
    if rates["structure"] == "tiered":
        if dwell_hours <= 3:
            return rates["first_3hr_flat"]
        elif dwell_hours <= 7:
            extra_hours = math.ceil(dwell_hours - 3)
            return rates["first_3hr_flat"] + extra_hours * rates["4th_to_7th_hr"]
        else:
            # Determine AM/PM rate based on entry time
            entry_hour = entry_time.hour
            rate_key = "7th_hr_plus_am" if 6 <= entry_hour < 12 else "7th_hr_plus_pm"
            extra_hours = math.ceil(dwell_hours - 7)
            return (rates["first_3hr_flat"] 
                    + 4 * rates["4th_to_7th_hr"] 
                    + extra_hours * rates.get(rate_key, rates.get("succeeding_hr_weekday", 20)))
```

---

#### [NEW] `leakage_detector.py`

Revenue recovery analysis using existing FSM state data.

| Detection Type | How It Works | Data Source |
|---------------|-------------|-------------|
| **Overstay** | `current_state.status == 'occupied_unpaid'` AND `dwell > threshold` | FSM timestamps |
| **Unauthorized** | Plate read exists but doesn't match ANY active ticket (matcher score < 0.80) | `matcher.py` results |
| **Validation Abuse** | `payment_settled_at` exists but vehicle still present hours later | Ticketing + state data |

**Revenue recovery calculation:**

```python
def compute_overstay_leakage(threshold_hours=4.0):
    """
    Data Provenance:
        - Dwell time: Tier B (entry_time from ticketing_records)
        - Hourly rate: Tier A (actual Megaworld rates)
        - Leakage benchmarks: Tier D (industry 5-15% range, cited)
    """
    overstaying_vehicles = get_vehicles_exceeding(threshold_hours)
    total_unpaid = sum(
        compute_ticket_revenue(v.entry_time, now, v.site, v.zone_type) 
        - v.amount_paid  # (₱0 if unpaid)
        for v in overstaying_vehicles
    )
    return {
        "total_unpaid_revenue": total_unpaid,
        "num_overstayers": len(overstaying_vehicles),
        "provenance": "Tier A rates × Tier B dwell times",
        "industry_context": "Industry average leakage: 5-15% (Tier D)"
    }
```

---

### Schema Enhancement

#### [MODIFY] [generate_data.py](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/generate_data.py)

**Enhance** the `ticketing_records` to include dwell-time diversity that enables realistic revenue computation:

- Currently: `entry_time` is random 10–180 minutes ago, `payment_settled_at` is 5–90 min after entry
- **Enhancement**: Model dwell time distributions by zone type:
  - **Mall**: Lognormal distribution centered at ~2.5 hours (captures quick-trip, standard, and leisure shoppers)
  - **Office**: Bimodal — 8–9 hour full-day parkers + 1–2 hour meeting visitors
  - **Residential**: 12–18 hour overnight parkers + 2–4 hour daytime visitors
- Add more tickets across the 28-day history (not just "right now") to enable daily/weekly revenue aggregation

This is the **most important change** — it gives us realistic per-ticket dwell times to compute revenue from.

---

### Dashboard Integration

#### [MODIFY] [app.py](file:///c:/Users/Tedd/Documents/College/2nd%20year/OJT/Megaworld/Personal%20Project/parking-poc/app.py)

Add **Tab 7: 💰 Revenue Intelligence** with 4 sections:

##### Section A: Revenue Dashboard
- **Township Revenue KPI Cards**: Total Revenue (today), Avg RPBH, Peak Revenue Hour, Revenue per Township
- **Revenue Heatmap**: Plotly heatmap — Zone × Hour colored by revenue intensity (₱)
- **Provenance Footer**: "Revenue computed from actual Megaworld parking rates (Tier A) × simulated occupancy data (Tier B)"

##### Section B: Dynamic Pricing Simulator
- **3-Strategy Comparison Chart**: Plotly multi-line showing projected daily revenue under:
  1. Current flat rate (Tier A actual rates)
  2. Time-of-day tiered (peak/off-peak split derived from occupancy curves)
  3. AI demand-responsive (ML-predicted occupancy → pricing function)
- **Interactive Sliders**: Base Rate, Target Occupancy, Surge Coefficient, Discount Floor
- **Uplift Banner**: "Dynamic pricing projects +₱X/day (+Y%) revenue uplift"
- **Industry Context Card**: "Industry case studies show 20–40% average uplift (Tier D: HAH Parking, SFpark)"

##### Section C: Revenue Leakage Recovery
- **Overstay Detection Panel**: List of flagged vehicles with computed unpaid revenue
- **Leakage KPIs**: Total Estimated Leakage (₱), Leakage Rate (%), Top Leaking Zones
- **Recovery Projection**: "Automated enforcement could recover ₱X/month"
- **Industry Benchmark**: "Manual facilities typically leak 5–15% of revenue (Tier D)"

##### Section D: Retail Intelligence
- **Dwell Time Distribution**: Plotly histogram by zone type with bucket labels
- **Visitor Value Score**: Estimated retail spend per visitor based on dwell time × PH benchmark
- **Validation Strategy Comparison**: Revenue impact of different free-parking-hour policies
- **Leasing Evidence Card**: "This zone sees X vehicles/day with Y avg dwell → estimated ₱Z daily retail spend"

---

## Resolved Architecture & Design Decisions

> [!NOTE]
> All prior open questions have been finalized and approved based on stakeholder feedback:

1. **Dual Historical Data Strategy (Approved Option C)**:
   - Use **derived revenue from `occupancy_history`** (interval occupancy × rate) for high-speed macro revenue heatmaps, 28-day revenue timelines, and diurnal aggregate trends.
   - Generate **realistic ticket sessions with archetype dwell distributions** across the 28-day window in `generate_data.py` (lognormal dwell for malls, bimodal for offices, multi-hour/overnight for residential) to power granular dwell-time histograms, leakage overstay detection, and visitor value profiling.

2. **Township Rate Structure Depth (Exact Fidelity)**:
   - Model all three townships (**Uptown Bonifacio**, **Eastwood City**, and **McKinley Hill / Venice Grand Canal Mall**) with their **exact, unique rate tariffs** (e.g., Eastwood's flat ₱60 weekend/holiday rate, Uptown's AM vs. PM 7th-hour escalation, overnight surcharges). Ensures 100% real-world credibility.

3. **Retail Spend Estimation & Calibration (Credible Range + Interactive Slider)**:
   - Ground the baseline in established Philippine retail research (Colliers Philippines / industry survey benchmark of **₱1,000 to ₱3,000 per mall visit**).
   - In the dashboard UI (Tab 7 Section D), present this as an interactive sensitivity slider allowing operators to adjust the baseline average spend (default ₱2,000, bounds ₱1,000–₱3,000) with real-time confidence bands and spend elasticity updates.

---

## Methodology Transparency Section

The tab itself should include an expandable **"📐 How These Numbers Were Calculated"** section that shows:

1. **Rate Sources**: "Parking rates sourced from Megaworld Lifestyle Malls official signage and MoneyMax.ph (2024–2025)"
2. **Revenue Formula**: The exact calculation: `Revenue = Σ(compute_ticket_revenue(entry, exit, site, zone))`
3. **Occupancy Basis**: "Occupancy patterns modeled on mathematically rigorous diurnal curves per zone type (office, mall, residential) with Philippine holiday and Megaworld event multipliers"
4. **Industry Citations**: Full citation list for dynamic pricing, leakage, dwell-time research
5. **Limitations**: "This is a POC simulation. Revenue figures represent estimates based on realistic synthetic occupancy applied to actual rate structures. Production deployment would use real gate telemetry."

This section ensures no one can accuse the numbers of being made up — every figure has a transparent calculation chain.

---

## Verification Plan

### Automated Tests
- `python -c "from revenue_config import PARKING_RATES; print('Rate config loaded')"` — verify rate structures parse correctly
- `python -c "from revenue_engine import compute_ticket_revenue; print(compute_ticket_revenue(...))"` — verify revenue calculation matches hand-computed example
- `streamlit run app.py` — verify Tab 7 renders without errors

### Manual Verification
- **Hand-calculate** one ticket's revenue using the Uptown rate card and verify it matches the function output
- **Cross-check** daily revenue aggregates against occupancy_history × average rate to ensure consistency
- **Verify** that all provenance badges are correctly displayed
- **Confirm** that the methodology section accurately describes the computation chain

### Sanity Checks
- Revenue per bay per hour should be in the ₱5–₱30 range (realistic for PH mall parking)
- Daily revenue per township should be in the ₱15,000–₱60,000 range (264 bays × realistic utilization)
- Leakage rate should fall within the 5–15% industry benchmark range
- Dynamic pricing uplift should be within the 15–40% documented range
