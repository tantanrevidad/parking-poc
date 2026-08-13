"""
app.py — Smart Parking POC
---------------------------
Single Streamlit app demonstrating every layer of the design end to end on
synthetic data:

  Tab 1: Live Seat Map                 — colored slot grid, driven by a simulated clock
  Tab 2: Trip Planner                  — baseline heuristic vs. trained ML prediction
  Tab 3: ML Model Insights             — real train/holdout metrics, feature importance
  Tab 4: Plate Matching Demo           — confidence-weighted matcher, step by step
  Tab 5: How the System Senses Parking — CV feasibility demo gallery

Run with:  streamlit run app.py
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

import generate_data as gd
import simulate
import predictor
import state_machine as sm

st.set_page_config(page_title="Smart Parking POC", layout="wide", page_icon="🅿️")

DB_PATH = "data/parking.db"
CV_DEMO_DIR = Path("cv-demo")


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

def ensure_data_exists():
    if not os.path.exists(DB_PATH):
        gd.main()


@st.cache_data(show_spinner=False)
def load_static_config():
    return simulate.load_static_config()


@st.cache_data(show_spinner=False)
def load_history_df():
    return predictor.load_history()


@st.cache_resource(show_spinner="Training gradient-boosted availability model...")
def get_trained_model(_history_df):
    model, metrics, importance_df, holdout = predictor.train_model(_history_df)
    baseline_lookup = predictor.baseline_heuristic(_history_df)
    return model, metrics, importance_df, holdout, baseline_lookup


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

ensure_data_exists()
sites_df, zones_df, slots_df, holidays, events = load_static_config()
history_df = load_history_df()
model, metrics, importance_df, holdout_df, baseline_lookup = get_trained_model(history_df)

if "sim_time" not in st.session_state:
    st.session_state.sim_time = datetime.now().replace(hour=18, minute=30, second=0, microsecond=0)

st.title("🅿️ Smart Parking Availability & Prediction — Proof of Concept")
st.caption(
    "All data below is synthetically generated (synthetic occupancy history, synthetic plates, "
    "synthetic ticketing records). The matching algorithm and the ML model are real and computed "
    "from that data — nothing here is hardcoded."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Controls")
    st.write(f"**Simulated time:** {st.session_state.sim_time.strftime('%a %b %d, %I:%M %p')}")

    c1, c2 = st.columns(2)
    if c1.button("⏩ +15 min"):
        st.session_state.sim_time += timedelta(minutes=15)
    if c2.button("⏩ +1 hour"):
        st.session_state.sim_time += timedelta(hours=1)

    if st.button("↺ Reset to evening peak (6:30 PM)"):
        st.session_state.sim_time = datetime.now().replace(hour=18, minute=30, second=0, microsecond=0)

    st.divider()

    # Site filter
    site_names = ["All Sites"] + list(sites_df["name"])
    selected_site = st.selectbox("🏢 Filter by township", options=site_names)

    st.divider()
    st.caption(
        f"Dataset: {len(sites_df)} sites · {len(slots_df)} slots across {len(zones_df)} zones · "
        f"{len(history_df):,} historical readings"
    )
    if st.button("🔄 Regenerate synthetic dataset"):
        gd.main()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Determine which zones are active based on the site filter
if selected_site == "All Sites":
    active_site_ids = list(sites_df.site_id)
else:
    active_site_ids = list(sites_df[sites_df.name == selected_site].site_id)

active_zones = zones_df[zones_df.site_id.isin(active_site_ids)]

# Recompute live state for the current simulated time
live_df = simulate.simulate_current_state(
    st.session_state.sim_time, zones_df, slots_df, holidays, events, sites_df
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Live Seat Map", "🚗 Trip Planner", "📊 ML Model Insights",
     "🔍 Plate Matching Demo", "👁️ How the System Senses Parking"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Live Seat Map
# ---------------------------------------------------------------------------

with tab1:
    legend_cols = st.columns(4)
    for i, status in enumerate([sm.FREE, sm.OCCUPIED_UNPAID, sm.OCCUPIED_PENDING_MATCH, sm.OCCUPIED_LIKELY_VACATING]):
        legend_cols[i].markdown(
            f"<span style='background-color:{sm.STATUS_COLORS[status]}; "
            f"padding:4px 10px; border-radius:4px;'>{sm.STATUS_LABELS[status]}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Group zones by site
    for _, site in sites_df.iterrows():
        if site.site_id not in active_site_ids:
            continue

        st.markdown(f"### 🏢 {site['name']}")
        site_zones = zones_df[zones_df.site_id == site.site_id]

        for _, z in site_zones.iterrows():
            zone_slots = live_df[live_df.zone_id == z.zone_id].sort_values("slot_code")
            n_free = (zone_slots.status == sm.FREE).sum()
            zone_type_badge = {"office": "🏢", "mall": "🛒", "residential": "🏠"}.get(z.zone_type, "📍")
            st.subheader(f"{zone_type_badge} {z.level} — {z.label}  ·  {n_free}/{len(zone_slots)} free")

            cols = st.columns(8)
            for i, (_, row) in enumerate(zone_slots.iterrows()):
                with cols[i % 8]:
                    color = sm.STATUS_COLORS[row.status]
                    st.markdown(
                        f"<div style='background-color:{color}; border-radius:6px; padding:10px 4px; "
                        f"text-align:center; margin-bottom:8px; color:#1a1a1a; font-weight:600; font-size:0.85em;'>"
                        f"{row.slot_code}</div>",
                        unsafe_allow_html=True,
                    )
            st.write("")

        st.divider()

# ---------------------------------------------------------------------------
# TAB 2 — Trip Planner
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Will there be a spot when I arrive?")
    col1, col2 = st.columns(2)
    with col1:
        zone_options = active_zones.zone_id.tolist()
        zone_choice = st.selectbox(
            "Zone", options=zone_options,
            format_func=lambda zid: (
                zones_df.loc[zones_df.zone_id == zid, "label"].iloc[0]
                + " — " + zones_df.loc[zones_df.zone_id == zid, "level"].iloc[0]
                + " (" + sites_df.loc[
                    sites_df.site_id == zones_df.loc[zones_df.zone_id == zid, "site_id"].iloc[0], "name"
                ].iloc[0] + ")"
            ),
        )
    with col2:
        hours_ahead = st.slider("Hours from now (simulated)", 0.0, 12.0, 2.0, step=0.5)

    target_ts = pd.Timestamp(st.session_state.sim_time + timedelta(hours=hours_ahead))
    result = predictor.predict_for_timestamp(model, baseline_lookup, history_df, zone_choice, target_ts)

    st.markdown(f"### Estimated availability at **{target_ts.strftime('%a %I:%M %p')}**")

    label_colors = {
        "Likely available": "#2ecc71",
        "Uncertain — may be tight": "#f1c40f",
        "Unlikely to have space": "#e74c3c",
    }
    st.markdown(
        f"<div style='background-color:{label_colors[result['label']]}; padding:16px; "
        f"border-radius:8px; font-size:1.3em; font-weight:700; color:#1a1a1a; text-align:center;'>"
        f"{result['label']}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline heuristic (occupancy rate)", f"{result['baseline_estimate']*100:.0f}%")
    m2.metric("Trained ML model (occupancy rate)", f"{result['trained_estimate']*100:.0f}%")
    m3.metric("Adjusted (conservative)", f"{result['adjusted_estimate']*100:.0f}%")

    st.caption(
        "The adjusted estimate nudges borderline predictions toward 'occupied' — an optimistic "
        "wrong prediction (telling someone a spot is free when it isn't) is worse than the reverse."
    )

    # Supporting chart: historical pattern for this zone/day-of-week
    dow = target_ts.dayofweek
    hist_feat = predictor.engineer_features(history_df)
    same_day = hist_feat[(hist_feat.zone_id == zone_choice) & (hist_feat.day_of_week == dow)]
    hourly = same_day.groupby("hour")["occupancy_rate"].mean().reset_index()
    fig = px.line(
        hourly, x="hour", y="occupancy_rate",
        title=f"Historical average occupancy — {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]}, this zone",
        markers=True,
    )
    fig.add_vline(x=target_ts.hour + target_ts.minute / 60, line_dash="dash", line_color="red")
    fig.update_yaxes(range=[0, 1], title="Occupancy rate")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — ML Model Insights
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Model performance (real, computed on a time-based holdout)")
    st.caption(
        "Train/test split is time-based (train on earlier weeks, evaluate on the most recent "
        "period) — not a random shuffle, which would leak future patterns into training."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trained model MAE", metrics["mae_trained_model"])
    m2.metric("Baseline heuristic MAE", metrics["mae_baseline"])
    m3.metric("Improvement over baseline", f"{metrics['improvement_pct']}%")
    m4.metric("Holdout rows", f"{metrics['n_holdout_rows']:,}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature importance** (permutation importance)")
        fig_imp = px.bar(
            importance_df, x="importance", y="feature", orientation="h",
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown("**Actual vs. predicted** (holdout sample)")
        # Use the first available zone_id instead of hardcoded 1
        available_zones = sorted(holdout_df.zone_id.unique())
        insight_zone = st.selectbox(
            "Select zone for comparison chart",
            options=available_zones,
            format_func=lambda zid: (
                zones_df.loc[zones_df.zone_id == zid, "label"].iloc[0]
                + " — " + zones_df.loc[zones_df.zone_id == zid, "level"].iloc[0]
            ) if zid in zones_df.zone_id.values else f"Zone {zid}",
        )
        sample = holdout_df[holdout_df.zone_id == insight_zone].tail(150)
        fig_pred = px.line(
            sample, x="ts", y=["occupancy_rate", "predicted_trained", "predicted_baseline"],
            labels={"value": "Occupancy rate", "ts": "Time"},
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    st.info(
        "Model: scikit-learn `HistGradientBoostingRegressor` — a gradient-boosted tree ensemble, "
        "chosen because this is tabular, seasonal data (time-of-day / day-of-week patterns), "
        "where boosted trees typically outperform deep learning and are far easier to explain."
    )

# ---------------------------------------------------------------------------
# TAB 4 — Plate Matching Demo
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Confidence-weighted plate-to-ticket matching")
    st.caption(
        "Pick a currently-occupied slot below to see its (synthetic, noisy) OCR read matched "
        "against the pool of paid-but-unresolved tickets in its zone — using the real matching "
        "algorithm, not a lookup."
    )

    candidates = live_df[live_df.status.isin([sm.OCCUPIED_UNPAID, sm.OCCUPIED_LIKELY_VACATING])]
    if candidates.empty:
        st.write("No occupied slots with a plate read at this simulated moment — advance the clock.")
    else:
        slot_choice = st.selectbox(
            "Slot", options=candidates.slot_id,
            format_func=lambda sid: candidates.loc[candidates.slot_id == sid, "slot_code"].iloc[0],
        )
        row = candidates[candidates.slot_id == slot_choice].iloc[0]

        c1, c2 = st.columns(2)
        c1.metric("OCR read (noisy)", row.read_text)
        c2.metric("True plate (for demo transparency)", row.plate)

        conf_df = pd.DataFrame({
            "character": list(row.read_text),
            "confidence": row.confidences,
        })
        st.markdown("**Per-character OCR confidence**")
        st.bar_chart(conf_df.set_index("character"))

        zone_pool = live_df[
            (live_df.zone_id == row.zone_id) & (live_df.ticket_id.notna())
        ][["ticket_id", "plate"]].drop_duplicates()
        pool_list = zone_pool.to_dict("records")

        import matcher as matcher_module
        result = matcher_module.match_plate(row.read_text, row.confidences, pool_list)

        st.markdown("**Candidate pool scoring** (higher = more similar)")
        ranked = pd.DataFrame(result["ranked_candidates"], columns=["ticket_id", "plate", "score"])
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        if result["resolved"]:
            st.success(f"✅ Matched to **{result['matched_ticket_id']}** — accepted above threshold with a clear margin.")
        else:
            st.warning(
                "⚠️ Left **unresolved** — either no candidate cleared the confidence threshold, "
                "or the top two candidates were too close to call. The slot stays "
                "'Occupied — Unpaid' rather than guessing."
            )

# ---------------------------------------------------------------------------
# TAB 5 — How the System Senses Parking (CV Feasibility Demo)
# ---------------------------------------------------------------------------

with tab5:
    st.subheader("How the System Senses Parking — Computer Vision Feasibility")
    st.caption(
        "This tab shows results from a standalone CV feasibility demo. Detection and OCR are run "
        "on generic public / synthetic images — NOT Megaworld property photos. The purpose is to "
        "demonstrate that vehicle detection and plate reading can be done cheaply with off-the-shelf "
        "models (YOLOv8n + Tesseract OCR)."
    )

    results_path = CV_DEMO_DIR / "matching_results.json"
    annotated_dir = CV_DEMO_DIR / "annotated"

    if not results_path.exists():
        st.info(
            "⚠️ No CV demo results found. Run `python cv_demo.py` to generate the feasibility "
            "demo output, then refresh this page."
        )
        st.code("python cv_demo.py", language="bash")
    else:
        with open(results_path) as f:
            cv_results = json.load(f)

        mode = cv_results[0].get("mode", "unknown") if cv_results else "unknown"
        st.markdown(f"**Mode:** `{mode}` — {'real YOLOv8n + Tesseract OCR' if mode == 'real' else 'simulated detection (demonstrates matching pipeline)'}")

        for i, result in enumerate(cv_results):
            st.markdown(f"---")
            st.markdown(f"### Scenario {i+1}: {result.get('notes', 'Detection result')}")

            col_img, col_data = st.columns([1, 1])

            with col_img:
                img_path = annotated_dir / result["image_file"]
                if img_path.exists():
                    st.image(str(img_path), caption=result["image_file"], use_container_width=True)
                else:
                    st.write(f"Image not found: {result['image_file']}")

            with col_data:
                st.metric("Vehicles detected", result["detection_count"])

                if result.get("plates_read"):
                    for plate in result["plates_read"]:
                        st.markdown(f"**OCR Read:** `{plate['ocr_text']}`  |  "
                                    f"**Avg Confidence:** {plate.get('confidence_avg', 'N/A')}")
                        if plate.get("char_confidences"):
                            conf_df = pd.DataFrame({
                                "char": list(plate["ocr_text"]),
                                "confidence": plate["char_confidences"][:len(plate["ocr_text"])],
                            })
                            st.bar_chart(conf_df.set_index("char"))

                match = result.get("matching_result", {})
                if match:
                    if match.get("resolved"):
                        st.success(f"✅ Matched to ticket **{match['matched_ticket_id']}**")
                    else:
                        st.warning("⚠️ Left **unresolved** — below threshold or near-tie")

                    if match.get("ranked_candidates"):
                        ranked = pd.DataFrame(match["ranked_candidates"],
                                              columns=["ticket_id", "plate", "score"])
                        st.dataframe(ranked, use_container_width=True, hide_index=True)

        st.divider()
        st.info(
            "**What this proves:** Vehicle detection (counting occupied/free spots) and license plate "
            "reading (for matching against ticketing records) are feasible with commodity hardware "
            "and off-the-shelf open-source models — no custom training required for the initial "
            "deployment. The existing confidence-weighted matching algorithm (`matcher.py`) handles "
            "noisy OCR output gracefully, as demonstrated above."
        )
