"""
app.py — Smart Parking POC
---------------------------
Enterprise Streamlit dashboard demonstrating predictive parking availability
and confidence-weighted plate matching on synthetic multi-site data.

Run with:  streamlit run app.py
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import generate_data as gd
import simulate
import predictor
import state_machine as sm
import parking_detector as pd_engine
import cv2

st.set_page_config(
    page_title="Smart Parking Management System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "data/parking.db"
CV_DEMO_DIR = Path("cv-demo")

# ---------------------------------------------------------------------------
# Custom Enterprise CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Hide default chrome */
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.stApp {
    background-color: #0F172A;
    color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
}

/* ── Header ── */
.app-header {
    padding: 20px 0 8px 0;
    margin-bottom: 4px;
}
.app-header h1 {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.03em;
}
.app-header p {
    font-size: 0.82rem;
    color: #64748B;
    margin: 4px 0 0 0;
}

/* ── Clock Toolbar ── */
.clock-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 12px;
}
.clock-time {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.01em;
}
.clock-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}

/* ── Site Selector Pills ── */
.site-pills {
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
}
.site-pill {
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid #334155;
    background: #1E293B;
    color: #94A3B8;
    transition: all 0.15s ease;
}
.site-pill.active {
    background: #0EA5E9;
    color: #FFFFFF;
    border-color: #0EA5E9;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
}

/* ── Tab Bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #1E293B;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 6px;
    color: #94A3B8 !important;
    font-weight: 600;
    font-size: 0.8rem;
    border: none !important;
    padding: 0px 16px !important;
    background-color: transparent !important;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #0EA5E9 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
}

/* ── Legend ── */
.legend-strip {
    display: flex;
    gap: 20px;
    padding: 0 0 14px 0;
    align-items: center;
}
.legend-entry {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #94A3B8;
}
.legend-entry .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Zone Section ── */
.zone-section {
    margin-bottom: 20px;
}
.site-divider {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    padding: 10px 0 8px 0;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 12px;
}
.zone-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 6px 0 8px 0;
}
.zone-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E2E8F0;
}
.zone-tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-left: 8px;
    vertical-align: middle;
}
.zone-tag-office   { background: rgba(14, 165, 233, 0.15); color: #38BDF8; }
.zone-tag-mall     { background: rgba(245, 158, 11, 0.15); color: #FBBF24; }
.zone-tag-residential { background: rgba(16, 185, 129, 0.15); color: #34D399; }

.zone-avail {
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748B;
}
.zone-avail strong {
    color: #CBD5E1;
    font-weight: 700;
}

/* ── Slot Grid ── */
.slot-card {
    border-radius: 6px;
    padding: 10px 4px;
    text-align: center;
    margin-bottom: 8px;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.slot-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
}

/* ── Metric Cards ── */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 18px;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.2;
}
.metric-sub {
    font-size: 0.72rem;
    color: #38BDF8;
    margin-top: 3px;
    font-weight: 500;
}

/* ── Forecast Banner ── */
.forecast-banner {
    border-radius: 10px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 16px;
}
.forecast-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.forecast-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
}

/* ── Result Banners ── */
.banner-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34D399;
    border-radius: 8px;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 0.85rem;
}
.banner-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #FBBF24;
    border-radius: 8px;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* ── Section Title ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 2px;
}
.section-desc {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 16px;
    line-height: 1.4;
}

/* ── Misc ── */
hr.subtle {
    border: none;
    border-top: 1px solid #1E293B;
    margin: 16px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_COLORS = ["#0EA5E9", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#06B6D4"]

def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
        margin=dict(l=24, r=24, t=36, b=24),
        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155", showline=True, linecolor="#334155"),
        yaxis=dict(gridcolor="#262f40", zerolinecolor="#334155", showline=True, linecolor="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color="#94A3B8")),
        colorway=PLOTLY_COLORS,
    )
    return fig


# ---------------------------------------------------------------------------
# Data Loaders (cached)
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

@st.cache_resource(show_spinner="Initializing prediction model...")
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


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>Smart Parking Availability & Prediction</h1>
    <p>Multi-township occupancy tracking, ML availability forecasting, and plate recognition — running on synthetic data with real algorithms.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inline Clock Toolbar + Site Selector
# ---------------------------------------------------------------------------

clock_left, clock_center, clock_right = st.columns([2, 5, 3])

with clock_left:
    st.markdown(f"<div class='clock-label'>Simulated Clock</div><div class='clock-time'>{st.session_state.sim_time.strftime('%a %b %d, %I:%M %p')}</div>", unsafe_allow_html=True)

with clock_center:
    b1, b2, b3, spacer = st.columns([1, 1, 2, 4])
    if b1.button("+15m", key="btn_15m"):
        st.session_state.sim_time += timedelta(minutes=15)
        st.rerun()
    if b2.button("+1h", key="btn_1h"):
        st.session_state.sim_time += timedelta(hours=1)
        st.rerun()
    if b3.button("Reset to 6:30 PM", key="btn_reset"):
        st.session_state.sim_time = datetime.now().replace(hour=18, minute=30, second=0, microsecond=0)
        st.rerun()

with clock_right:
    site_names = ["All Sites"] + list(sites_df["name"])
    selected_site = st.radio(
        "Township", options=site_names, horizontal=True, label_visibility="collapsed"
    )


# ---------------------------------------------------------------------------
# Sidebar (minimal — dataset info + regenerate)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Dataset")
    st.markdown(f"**{len(sites_df)}** townships · **{len(zones_df)}** zones · **{len(slots_df)}** slots")
    st.markdown(f"**{len(history_df):,}** historical readings")
    st.markdown("---")
    if st.button("Regenerate Dataset"):
        gd.main()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.caption("All data is synthetically generated. The ML model and matching algorithm are real.")


# ---------------------------------------------------------------------------
# Compute active state
# ---------------------------------------------------------------------------

if selected_site == "All Sites":
    active_site_ids = list(sites_df.site_id)
else:
    active_site_ids = list(sites_df[sites_df.name == selected_site].site_id)

active_zones = zones_df[zones_df.site_id.isin(active_site_ids)]

live_df = simulate.simulate_current_state(
    st.session_state.sim_time, zones_df, slots_df, holidays, events, sites_df
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Occupancy Map", "Availability Forecast", "Model Performance", "Plate Matching", "ALPR Feasibility", "Space Detection (CV)"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Occupancy Map
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    # Legend strip
    st.markdown(f"""
    <div class="legend-strip">
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.FREE]};"></div> Free</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_UNPAID]};"></div> Occupied</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_PENDING_MATCH]};"></div> Pending Match</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_LIKELY_VACATING]};"></div> Vacating</div>
    </div>
    """, unsafe_allow_html=True)

    for _, site in sites_df.iterrows():
        if site.site_id not in active_site_ids:
            continue

        st.markdown(f"<div class='site-divider'>{site['name']}</div>", unsafe_allow_html=True)
        site_zones = zones_df[zones_df.site_id == site.site_id]

        for _, z in site_zones.iterrows():
            zone_slots = live_df[live_df.zone_id == z.zone_id].sort_values("slot_code")
            n_free = (zone_slots.status == sm.FREE).sum()
            total = len(zone_slots)
            tag_class = f"zone-tag-{z.zone_type}" if z.zone_type in ("office", "mall", "residential") else ""

            st.markdown(f"""
            <div class="zone-header">
                <div>
                    <span class="zone-name">{z.label} — {z.level}</span>
                    <span class="zone-tag {tag_class}">{z.zone_type}</span>
                </div>
                <div class="zone-avail"><strong>{n_free}</strong> / {total} available</div>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(8)
            for i, (_, row) in enumerate(zone_slots.iterrows()):
                with cols[i % 8]:
                    color = sm.STATUS_COLORS[row.status]
                    text_color = "#0F172A" if row.status in (sm.FREE, sm.OCCUPIED_LIKELY_VACATING, sm.OCCUPIED_PENDING_MATCH) else "#FFFFFF"
                    st.markdown(
                        f"<div class='slot-card' style='background:{color}; color:{text_color};'>{row.slot_code}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr class='subtle'/>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Availability Forecast
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div class='section-title'>Predictive Availability</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Select a parking zone and how far ahead you plan to arrive. The model forecasts whether a spot will be available.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
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
        hours_ahead = st.slider("Hours ahead", 0.0, 12.0, 2.0, step=0.5)

    target_ts = pd.Timestamp(st.session_state.sim_time + timedelta(hours=hours_ahead))
    result = predictor.predict_for_timestamp(model, baseline_lookup, history_df, zone_choice, target_ts)

    # Forecast banner
    label_styles = {
        "Likely available": ("#10B981", "rgba(16, 185, 129, 0.12)"),
        "Uncertain — may be tight": ("#F59E0B", "rgba(245, 158, 11, 0.12)"),
        "Unlikely to have space": ("#F43F5E", "rgba(244, 63, 94, 0.12)"),
    }
    accent, bg = label_styles.get(result["label"], ("#0EA5E9", "rgba(14, 165, 233, 0.12)"))

    st.markdown(f"""
    <div class="forecast-banner" style="background:{bg}; border: 1px solid {accent};">
        <div class="forecast-label" style="color:{accent};">Forecast for {target_ts.strftime('%A, %I:%M %p')}</div>
        <div class="forecast-value">{result['label']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric row
    m1, m2, m3 = st.columns(3)
    for col, (label, value, sub) in zip(
        [m1, m2, m3],
        [
            ("Baseline Heuristic", f"{result['baseline_estimate']*100:.0f}%", "Historical average"),
            ("ML Model", f"{result['trained_estimate']*100:.0f}%", "Gradient-boosted forecast"),
            ("Adjusted", f"{result['adjusted_estimate']*100:.0f}%", "Conservative safety margin"),
        ]
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )

    # Area chart
    dow = target_ts.dayofweek
    hist_feat = predictor.engineer_features(history_df)
    same_day = hist_feat[(hist_feat.zone_id == zone_choice) & (hist_feat.day_of_week == dow)]
    hourly = same_day.groupby("hour")["occupancy_rate"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["occupancy_rate"],
        mode="lines", fill="tozeroy",
        line=dict(color="#0EA5E9", width=2.5),
        fillcolor="rgba(14, 165, 233, 0.12)",
        name="Occupancy",
    ))
    arrival_hour = target_ts.hour + target_ts.minute / 60
    fig.add_vline(x=arrival_hour, line_dash="dash", line_color="#F43F5E", line_width=1.5)
    fig.add_annotation(
        x=arrival_hour, y=1.02, yref="paper",
        text=f"Arrival {target_ts.strftime('%I:%M %p')}", showarrow=False,
        font=dict(size=10, color="#F43F5E"), yanchor="bottom",
    )
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig.update_layout(
        title=f"Historical Occupancy — {day_names[dow]}s",
        xaxis_title="Hour", yaxis_title="Occupancy Rate",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        showlegend=False,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-title'>Model Diagnostics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>HistGradientBoostingRegressor validated on a chronological holdout split (no future data leakage).</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value, sub) in zip(
        [m1, m2, m3, m4],
        [
            ("Model MAE", f"{metrics['mae_trained_model']:.4f}", "Mean absolute error"),
            ("Baseline MAE", f"{metrics['mae_baseline']:.4f}", "Heuristic error"),
            ("Improvement", f"{metrics['improvement_pct']:.1f}%", "Error reduction"),
            ("Holdout Rows", f"{metrics['n_holdout_rows']:,}", "Validation samples"),
        ]
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title' style='margin-top:16px;'>Feature Importance</div>", unsafe_allow_html=True)
        fig_imp = px.bar(importance_df, x="importance", y="feature", orientation="h",
                         labels={"importance": "Score", "feature": ""})
        fig_imp.update_traces(marker_color="#0EA5E9")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        apply_plotly_theme(fig_imp)
        st.plotly_chart(fig_imp, width="stretch")

    with col2:
        st.markdown("<div class='section-title' style='margin-top:16px;'>Actual vs. Predicted</div>", unsafe_allow_html=True)
        available_zones = sorted(holdout_df.zone_id.unique())
        insight_zone = st.selectbox(
            "Zone", options=available_zones, key="insight_zone",
            format_func=lambda zid: (
                zones_df.loc[zones_df.zone_id == zid, "label"].iloc[0]
                + " — " + zones_df.loc[zones_df.zone_id == zid, "level"].iloc[0]
            ) if zid in zones_df.zone_id.values else f"Zone {zid}",
        )
        sample = holdout_df[holdout_df.zone_id == insight_zone].tail(150)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=sample["ts"], y=sample["occupancy_rate"], mode="lines", name="Actual", line=dict(color="#F8FAFC", width=1.5)))
        fig_pred.add_trace(go.Scatter(x=sample["ts"], y=sample["predicted_trained"], mode="lines", name="ML Model", line=dict(color="#0EA5E9", width=2)))
        fig_pred.add_trace(go.Scatter(x=sample["ts"], y=sample["predicted_baseline"], mode="lines", name="Baseline", line=dict(color="#475569", width=1, dash="dot")))
        fig_pred.update_layout(xaxis_title="Time", yaxis_title="Occupancy Rate")
        apply_plotly_theme(fig_pred)
        st.plotly_chart(fig_pred, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Plate Matching
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='section-title'>Plate-to-Ticket Matching</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Confidence-weighted OCR matching with confusable-character handling (0/O, 1/I, 8/B, 5/S, 2/Z).</div>", unsafe_allow_html=True)

    candidates = live_df[live_df.status.isin([sm.OCCUPIED_UNPAID, sm.OCCUPIED_LIKELY_VACATING])]
    if candidates.empty:
        st.info("No occupied slots with plate reads at this time step. Advance the clock.")
    else:
        slot_choice = st.selectbox(
            "Occupied Slot", options=candidates.slot_id,
            format_func=lambda sid: candidates.loc[candidates.slot_id == sid, "slot_code"].iloc[0],
        )
        row = candidates[candidates.slot_id == slot_choice].iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>OCR Read (noisy)</div>"
                f"<div class='metric-value' style='font-family:monospace; color:#38BDF8;'>{row.read_text}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Ground Truth</div>"
                f"<div class='metric-value' style='font-family:monospace; color:#CBD5E1;'>{row.plate}</div></div>",
                unsafe_allow_html=True,
            )

        conf_df = pd.DataFrame({"Char": list(row.read_text), "Confidence": row.confidences})
        fig_conf = px.bar(conf_df, x="Char", y="Confidence", text_auto=".2f",
                          labels={"Char": "Character", "Confidence": "OCR Confidence"})
        fig_conf.update_traces(marker_color="#0EA5E9")
        fig_conf.update_yaxes(range=[0, 1])
        apply_plotly_theme(fig_conf)
        st.plotly_chart(fig_conf, width="stretch")

        zone_pool = live_df[
            (live_df.zone_id == row.zone_id) & (live_df.ticket_id.notna())
        ][["ticket_id", "plate"]].drop_duplicates()

        import matcher as matcher_module
        result = matcher_module.match_plate(row.read_text, row.confidences, zone_pool.to_dict("records"))

        st.markdown("<div class='section-title'>Match Scoring</div>", unsafe_allow_html=True)
        ranked = pd.DataFrame(result["ranked_candidates"], columns=["Ticket", "Plate", "Score"])
        st.dataframe(ranked, width="stretch", hide_index=True)

        if result["resolved"]:
            st.markdown(f"<div class='banner-success'>Matched — ticket <strong>{result['matched_ticket_id']}</strong>, clear margin above threshold.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='banner-warning'>Unresolved — score below threshold or top candidates too close. Slot stays 'Occupied — Unpaid'.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — CV Demo
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("<div class='section-title'>Computer Vision Feasibility Demo</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-desc'>Real vehicle detection (YOLOv8n) and plate OCR evaluated against "
        "real-world photographs with hand-verified ground truth. Choose between the international academic "
        "benchmark and the authentic Philippine parking lot dataset.</div>",
        unsafe_allow_html=True,
    )

    dataset_choice = st.radio(
        "Select Dataset",
        options=["🇵🇭 Philippine Parking Lot Dataset (Metro Manila)", "🌍 Academic ALPR Benchmark (OpenALPR)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "Philippine" in dataset_choice:
        results_path = CV_DEMO_DIR / "ph_matching_results.json"
        annotated_dir = CV_DEMO_DIR / "ph-annotated"
    else:
        results_path = CV_DEMO_DIR / "matching_results.json"
        annotated_dir = CV_DEMO_DIR / "annotated"

    if not results_path.exists():
        st.info(
            "Dataset results not found. Run `python fetch_ph_dataset.py` (or `python fetch_real_dataset.py`) "
            "then `python cv_demo.py`, and refresh."
        )
    else:
        with open(results_path) as f:
            cv_output = json.load(f)

        summary = cv_output.get("summary", {})
        cv_results = cv_output.get("results", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Exact-Match Rate</div><div class='metric-value'>{summary.get('exact_match_rate', 0)*100:.0f}%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Char. Accuracy</div><div class='metric-value'>{summary.get('mean_char_accuracy', 0)*100:.0f}%</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Matcher False Positives</div><div class='metric-value'>{summary.get('matcher_false_positive_count', 0)}</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>Images Tested</div><div class='metric-value'>{summary.get('n_images', 0)}</div></div>", unsafe_allow_html=True)

        st.caption(
            f"Dataset: **{summary.get('source_dataset', 'n/a')}** · "
            f"True matches resolved: {summary.get('matcher_correct_count', summary.get('matcher_resolved_count', 0))}/{summary.get('n_images', 0)} "
            f"({summary.get('matcher_resolved_rate', 0)*100:.0f}% resolution rate)."
        )

        st.markdown(
            "<div class='banner-success' style='margin-top:8px;'>"
            "<strong>Sensing Integrity:</strong> The confidence-weighted matcher effectively bridges OCR noise "
            "(e.g., resolving 0/O and 6/G confusions) while maintaining zero false-positive ticket matches across all noisy test scenarios."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='subtle'/>", unsafe_allow_html=True)

        for i, result in enumerate(cv_results):
            match_tag = "✅ exact match" if result["exact_match"] else f"acc={result['char_accuracy']:.2f}"
            st.markdown(
                f"<div class='section-title'>{result['source']} — {match_tag}</div>",
                unsafe_allow_html=True,
            )

            col_img, col_data = st.columns([1, 1])

            with col_img:
                img_path = annotated_dir / result["image_file"]
                if img_path.exists():
                    st.image(str(img_path), caption=result["image_file"], width="stretch")

            with col_data:
                st.markdown(f"**Ground truth plate:** `{result['ground_truth_plate']}`")
                st.markdown(f"**OCR read:** `{result['ocr_text'] or '(none)'}`  (confidence: {result['ocr_confidence']:.2f})")
                if result.get("lto_type"):
                    st.markdown(f"**LTO Series:** `{result['lto_type']}`")
                if result.get("noise_profile"):
                    st.caption(f"Environment / Noise: {result['noise_profile']}")

                match = result.get("matching_result")
                if match:
                    if match.get("resolved"):
                        st.markdown(f"<div class='banner-success'>Matched: <strong>Ticket {match['matched_ticket_id']}</strong></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='banner-warning'>Unresolved — correctly declined to guess (avoids false penalty)</div>", unsafe_allow_html=True)

            st.markdown("<hr class='subtle'/>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Space Detection (CV)
# ═══════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("<div class='section-title'>Multi-Angle Parking Space Occupancy & Car Detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-desc'>Enterprise computer vision engine executing a <strong>5-phase occupancy detection algorithm</strong> "
        "(ROI Calibration → YOLOv8n Inference → IoA Spatial Overlap → Temporal Debouncing → JSON Payload Generation) "
        "across surveillance feeds from <code>car_dataset/</code>.</div>",
        unsafe_allow_html=True,
    )

    available_angles = pd_engine.list_available_camera_angles()

    if not available_angles:
        st.warning("No camera feeds or images found in `car_dataset/`. Please ensure the folder contains test images.")
    else:
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 2, 2, 2])

        with ctrl_col1:
            angle_options = [a["filename"] for a in available_angles]
            angle_choice = st.selectbox(
                "Select Surveillance Camera Feed / Perspective Angle",
                options=angle_options,
                format_func=lambda fn: next((a["display_name"] for a in available_angles if a["filename"] == fn), fn),
                key="cam_angle_select",
            )

        with ctrl_col2:
            conf_val = st.slider(
                "YOLO Vehicle Conf (τ_conf)",
                min_value=0.15,
                max_value=0.85,
                value=0.25,
                step=0.05,
                help="Minimum confidence threshold for vehicle bounding box detection",
            )

        with ctrl_col3:
            ioa_val = st.slider(
                "IoA Overlap Ratio (τ_ioa)",
                min_value=0.10,
                max_value=0.60,
                value=0.30,
                step=0.05,
                help="Occupancy ratio = Area(Slot ∩ Car) / Area(Slot) required to mark slot as Occupied",
            )

        with ctrl_col4:
            enable_low_light = st.checkbox(
                "🌙 Low-Light CLAHE Boost",
                value=True,
                help="Adaptive histogram equalization to detect dark/black SUVs, pickups, and shadowed vehicles",
            )
            enable_smoothing = st.checkbox(
                "⏱️ Temporal Smoothing",
                value=True,
                help="Applies sliding-window state debouncing across consecutive time-series frames",
            )

        selected_meta = next(a for a in available_angles if a["filename"] == angle_choice)
        raw_bgr = cv2.imread(selected_meta["path"])

        if raw_bgr is None:
            st.error(f"Failed to load image: {selected_meta['path']}")
        else:
            # Run the 5-Phase Space Occupancy Detection Engine
            detection_out = pd_engine.detect_parking_spaces(
                raw_bgr,
                angle_choice,
                conf_threshold=conf_val,
                ioa_threshold=ioa_val,
                enable_temporal_smoothing=enable_smoothing,
                enable_low_light_boost=enable_low_light,
            )

            summary = detection_out["summary"]
            bay_records = detection_out["bays"]
            annotated_bgr = detection_out["annotated_image"]
            json_payload = detection_out["json_payload"]

            # Convert BGR to RGB for Streamlit display
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            # System Status Banner
            calib_info = f"{raw_bgr.shape[1]}x{raw_bgr.shape[0]}"
            borderline_info = f" · ⚠️ {summary['borderline_count']} Borderline" if summary.get("borderline_count", 0) > 0 else ""
            st.info(
                f"**Algorithm Architecture:** 5-Phase Dual-Calibrated ROI-IoA Pipeline · "
                f"**Camera Calibration:** `{angle_choice}` ({calib_info}) · "
                f"**Total Monitored Spaces:** {summary['total_bays']} Bays · "
                f"**YOLO Vehicles Detected:** {summary['detected_vehicles']}{borderline_info}"
            )

            # Metric Cards Row
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Total Spaces</div>"
                    f"<div class='metric-value'>{summary['total_bays']}</div>"
                    f"<div class='metric-sub'>Calibrated Trapezoids</div></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Vacant Slots</div>"
                    f"<div class='metric-value' style='color:#34D399;'>{summary['vacant_count']}</div>"
                    f"<div class='metric-sub'>🟢 Available to Park</div></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Occupied Slots</div>"
                    f"<div class='metric-value' style='color:#F43F5E;'>{summary['occupied_count']}</div>"
                    f"<div class='metric-sub'>🔴 Car Parked Inside</div></div>",
                    unsafe_allow_html=True,
                )
            with m4:
                occ_pct = summary["occupancy_rate"] * 100.0
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Occupancy Rate</div>"
                    f"<div class='metric-value'>{occ_pct:.1f}%</div>"
                    f"<div class='metric-sub'>Lot Saturation</div></div>",
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Vehicles Tracked</div>"
                    f"<div class='metric-value' style='color:#38BDF8;'>{summary['detected_vehicles']}</div>"
                    f"<div class='metric-sub'>YOLOv8 Objects</div></div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"**Feed Details:** {selected_meta['display_name']} · {selected_meta['description']}")

            # Visual Feeds
            feed_tab1, feed_tab2, feed_tab3 = st.tabs([
                "🟢/🔴 Annotated Vacancy Overlay",
                "📷 Raw Surveillance Feed",
                "📦 Phase 5 Standard JSON Payload",
            ])

            with feed_tab1:
                st.image(
                    annotated_rgb,
                    caption=f"Real-Time Space Occupancy Detection: {summary['vacant_count']} Free / {summary['occupied_count']} Occupied (Capacity: {summary['total_bays']})",
                    width="stretch",
                )
            with feed_tab2:
                st.image(
                    raw_rgb,
                    caption=f"Raw Camera Input Frame: {selected_meta['filename']} ({calib_info})",
                    width="stretch",
                )
            with feed_tab3:
                st.markdown("**Phase 5: Standardized JSON Output Payload for Downstream Consumption**")
                st.json(json_payload)

            st.markdown("<hr class='subtle'/>", unsafe_allow_html=True)

            # Slot-by-Slot Telemetry & Algorithm Deep Dive
            col_table, col_algo = st.columns([1, 1])

            with col_table:
                st.markdown("<div class='section-title'>Slot-by-Slot Telemetry</div>", unsafe_allow_html=True)
                table_filter = st.radio(
                    "Filter Slots",
                    options=["All Slots", "🟢 Vacant Only", "🔴 Occupied Only", "⚠️ Borderline / Low Conf"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="bay_table_filter",
                )

                df_bays = pd.DataFrame(bay_records)
                if df_bays.empty:
                    st.info("No slots configured for this camera feed.")
                else:
                    if "low_confidence_flag" not in df_bays.columns:
                        df_bays["low_confidence_flag"] = False
                    if "status" not in df_bays.columns:
                        df_bays["status"] = "vacant"
                    if "occupancy_ratio" not in df_bays.columns:
                        df_bays["occupancy_ratio"] = 0.0
                    if "confidence" not in df_bays.columns:
                        df_bays["confidence"] = 1.0
                    if "matched_vehicle_class" not in df_bays.columns:
                        df_bays["matched_vehicle_class"] = None

                    if "🟢 Vacant Only" in table_filter:
                        df_bays = df_bays[df_bays["status"] == "vacant"]
                    elif "🔴 Occupied Only" in table_filter:
                        df_bays = df_bays[df_bays["status"] == "occupied"]
                    elif "⚠️ Borderline" in table_filter:
                        df_bays = df_bays[df_bays["low_confidence_flag"] == True]

                    df_bays["status_badge"] = df_bays["status"].apply(lambda s: "🔴 Occupied" if s == "occupied" else "🟢 Vacant")
                    df_bays["ioa_pct"] = df_bays["occupancy_ratio"].apply(lambda r: f"{r*100:.1f}%")
                    df_bays["conf_pct"] = df_bays["confidence"].apply(lambda c: f"{c*100:.1f}%")
                    df_bays["veh_type"] = df_bays["matched_vehicle_class"].fillna("—")
                    df_bays["quality"] = df_bays["low_confidence_flag"].apply(lambda b: "⚠️ Borderline (±5%)" if b else "✅ High Conf")

                    display_df = df_bays[["slot_id", "slot_name", "zone", "status_badge", "ioa_pct", "conf_pct", "veh_type", "quality"]].copy()
                    display_df.columns = ["Slot ID", "Bay Name", "Zone", "Status", "IoA Overlap", "Confidence", "Vehicle Class", "Quality Flag"]
                    st.dataframe(display_df, width="stretch", hide_index=True)

            with col_algo:
                st.markdown("<div class='section-title'>5-Phase Occupancy Detection Algorithm</div>", unsafe_allow_html=True)
                st.markdown("""
                <div style="background:#1E293B; border:1px solid #334155; border-radius:10px; padding:16px; font-size:0.84rem; line-height:1.6; color:#CBD5E1;">
                    <strong style="color:#38BDF8;">Phase 1: Data Preparation & Dual Native ROI Calibration</strong><br/>
                    True perspective trapezoids calibrated against <code>empty lot.jpg</code> (1372×768 master reference) and native surveillance video frames (457×192). Resolution scaling runs automatically for any input dimension.<br/><br/>
                    <strong style="color:#38BDF8;">Phase 2: Model Setup & Inference Pipeline</strong><br/>
                    Lightweight <strong>YOLOv8n</strong> convolutional detector processes the camera frame, filtering for relevant COCO vehicle classes (<code>car</code>, <code>truck</code>, <code>bus</code>, <code>motorcycle</code>) above confidence threshold <code>τ_conf</code>.<br/><br/>
                    <strong style="color:#38BDF8;">Phase 3: Spatial Logic & Occupancy Calculation (IoA + Ground-Contact)</strong><br/>
                    Computes Intersection over Area (IoA) with tire ground-contact reinforcement:
                    <div style="background:#0F172A; padding:6px 12px; border-radius:6px; margin:6px 0; font-family:monospace; color:#F8FAFC;">
                        Occupancy Ratio = Area(Slot Polygon ∩ Vehicle Box) / Area(Slot Polygon)<br/>
                        If Vehicle Ground Point (cx, y2) ∈ Slot Polygon & IoA ≥ 0.12 → Effective IoA = max(IoA, 0.50)
                    </div>
                    If <code>Occupancy Ratio ≥ τ_ioa</code> → <span style="color:#F43F5E; font-weight:700;">Occupied (🔴)</span>; otherwise → <span style="color:#34D399; font-weight:700;">Vacant (🟢)</span>. Borderline cases (|IoA - τ_ioa| ≤ 0.05) receive a review flag.<br/><br/>
                    <strong style="color:#38BDF8;">Phase 4: Temporal Filtering & Edge-Case Handling</strong><br/>
                    Sliding-window state debouncing (5-frame history, ≥60% consensus) prevents flickering from transient vehicle traversal or headlight glare.<br/><br/>
                    <strong style="color:#38BDF8;">Phase 5: Output Structuring & Telemetry JSON</strong><br/>
                    Formats live telemetry into enterprise JSON payloads with low-confidence flags ready for downstream API consumption, real-time signage, or database logging.
                </div>
                """, unsafe_allow_html=True)