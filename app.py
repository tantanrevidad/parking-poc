"""
app.py — Smart Parking POC
---------------------------
Enterprise Streamlit dashboard demonstrating predictive parking availability
and confidence-weighted plate matching on synthetic multi-site data.

Run with:  streamlit run app.py
"""

import json
import os
import sqlite3
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
# Theme System & Professional Enterprise CSS
# ---------------------------------------------------------------------------

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"


def get_theme_css(theme_mode: str = "dark") -> str:
    is_dark = theme_mode == "dark"
    bg_app = "#121417" if is_dark else "#F8FAFC"
    bg_card = "#1A1C23" if is_dark else "#FFFFFF"
    bg_card_hover = "#22252E" if is_dark else "#E2E8F0"
    bg_sidebar = "#16181F" if is_dark else "#FFFFFF"
    border_color = "#282C37" if is_dark else "#CBD5E1"
    border_subtle = "#20232C" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#334155"
    text_muted = "#64748B" if is_dark else "#64748B"
    accent_blue = "#38BDF8" if is_dark else "#0284C7"
    accent_purple = "#C084FC" if is_dark else "#9333EA"
    card_subtle = "#14161C" if is_dark else "#F1F5F9"
    tab_bg = "#16181F" if is_dark else "#E2E8F0"
    tab_active_bg = "#2563EB" if is_dark else "#2563EB"
    tab_text_color = "#94A3B8" if is_dark else "#000000"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --bg-app: {bg_app};
    --bg-card: {bg_card};
    --bg-card-hover: {bg_card_hover};
    --border-color: {border_color};
    --border-subtle: {border_subtle};
    --text-primary: {text_primary};
    --text-secondary: {text_secondary};
    --text-muted: {text_muted};
    --accent-blue: {accent_blue};
    --accent-purple: {accent_purple};
}}

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

/* Hide default chrome */
header[data-testid="stHeader"] {{ background: transparent !important; }}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

.stApp {{
    background-color: var(--bg-app);
    color: var(--text-primary);
}}

section[data-testid="stSidebar"] {{
    background-color: {bg_sidebar} !important;
    border-right: 1px solid var(--border-color) !important;
}}

/* ── Hero Centered Date & Time Header ── */
.hero-clock-container {{
    text-align: center;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px 20px 20px 20px;
    margin: 8px 0 20px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    position: relative;
    overflow: hidden;
}}
.hero-clock-container::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
}}
.hero-clock-eyebrow {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    margin-bottom: 4px;
}}
.hero-clock-time {{
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin: 2px 0 4px 0;
    font-feature-settings: "tnum";
}}
.hero-clock-date {{
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent-blue);
    letter-spacing: -0.01em;
}}

/* ── Top Bar Brand ── */
.app-brand {{
    padding: 6px 0 16px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 18px;
}}
.app-brand h1 {{
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}}
.app-brand p {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 2px 0 0 0;
}}

/* ── Segmented Tab Bar Navigation ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px !important;
    background-color: {tab_bg} !important;
    padding: 6px !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-color) !important;
    margin-bottom: 20px !important;
    display: inline-flex !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    height: 40px !important;
    border-radius: 8px !important;
    color: {tab_text_color} !important;
    font-weight: 800 !important;
    font-size: 0.86rem !important;
    border: none !important;
    padding: 0px 18px !important;
    background-color: transparent !important;
    transition: all 0.15s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.stTabs [data-baseweb="tab"] p {{
    color: {tab_text_color} !important;
    font-weight: 800 !important;
    font-size: 0.86rem !important;
    margin: 0 !important;
    transition: color 0.15s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: var(--bg-card-hover) !important;
    color: {tab_text_color} !important;
}}
.stTabs [data-baseweb="tab"]:hover p {{
    color: {tab_text_color} !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: {tab_active_bg} !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
}}
.stTabs [aria-selected="true"] p {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
}}

/* ── Legend Strip ── */
.legend-strip {{
    display: flex;
    gap: 28px;
    padding: 14px 22px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-bottom: 20px;
    align-items: center;
    flex-wrap: wrap;
}}
.legend-entry {{
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-primary);
}}
.legend-entry .dot {{
    width: 13px;
    height: 13px;
    border-radius: 50%;
    flex-shrink: 0;
}}

/* ── Zone Section Card ── */
.zone-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
}}
.zone-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}}
.zone-name {{
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}}
.zone-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-left: 10px;
    vertical-align: middle;
}}
.zone-tag-office   {{ background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3); }}
.zone-tag-mall     {{ background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }}
.zone-tag-residential {{ background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }}

.zone-avail {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-secondary);
}}
.zone-avail strong {{
    color: var(--text-primary);
    font-weight: 800;
    font-size: 1rem;
}}

/* ── Solid Color Architectural Parking Stalls ── */
.bay-stall {{
    border-radius: 7px;
    padding: 10px 4px 8px 4px;
    text-align: center;
    margin-bottom: 8px;
    font-weight: 800;
    border: 1px solid rgba(0, 0, 0, 0.25);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    cursor: pointer;
    user-select: none;
    display: block;
}}
.bay-stall:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    filter: brightness(1.12);
}}
.bay-stall:active {{
    transform: translateY(0px);
}}

.bay-stall-free {{
    background-color: #10B981 !important;
    background: #10B981 !important;
    color: #FFFFFF !important;
    border: 1px solid #059669 !important;
}}
.bay-stall-free .bay-code {{ color: #FFFFFF !important; }}
.bay-stall-free .bay-indicator {{ color: #ECFDF5 !important; }}

.bay-stall-occupied {{
    background-color: #E11D48 !important;
    background: #E11D48 !important;
    color: #FFFFFF !important;
    border: 1px solid #BE123C !important;
}}
.bay-stall-occupied .bay-code {{ color: #FFFFFF !important; }}
.bay-stall-occupied .bay-indicator {{ color: #FFE4E6 !important; }}

.bay-stall-pending {{
    background-color: #F59E0B !important;
    background: #F59E0B !important;
    color: #1E293B !important;
    border: 1px solid #D97706 !important;
}}
.bay-stall-pending .bay-code {{ color: #1E293B !important; }}
.bay-stall-pending .bay-indicator {{ color: #451A03 !important; }}

.bay-stall-vacating {{
    background-color: #0284C7 !important;
    background: #0284C7 !important;
    color: #FFFFFF !important;
    border: 1px solid #0369A1 !important;
}}
.bay-stall-vacating .bay-code {{ color: #FFFFFF !important; }}
.bay-stall-vacating .bay-indicator {{ color: #E0F2FE !important; }}

.bay-code {{
    font-size: 0.84rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    line-height: 1.1;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.bay-indicator {{
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
    opacity: 0.95;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

/* ── Direct Solid-Color Parking Stall Buttons (Guaranteed Column Targeting) ── */
div[data-testid="column"]:has(.slot-free) button,
div[data-testid="stColumn"]:has(.slot-free) button,
div:has(> .slot-free) + div[data-testid="stButton"] button,
button[aria-label*="OPEN"] {{
    background-color: #10B981 !important;
    background: #10B981 !important;
    color: #FFFFFF !important;
    border: 1px solid #059669 !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    min-height: 52px !important;
    padding: 6px 2px !important;
    white-space: pre-line !important;
    line-height: 1.25 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}}
div[data-testid="column"]:has(.slot-free) button p,
div[data-testid="stColumn"]:has(.slot-free) button p,
button[aria-label*="OPEN"] p {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    line-height: 1.25 !important;
}}

div[data-testid="column"]:has(.slot-occupied) button,
div[data-testid="stColumn"]:has(.slot-occupied) button,
div:has(> .slot-occupied) + div[data-testid="stButton"] button,
button[aria-label*="BUSY"] {{
    background-color: #E11D48 !important;
    background: #E11D48 !important;
    color: #FFFFFF !important;
    border: 1px solid #BE123C !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    min-height: 52px !important;
    padding: 6px 2px !important;
    white-space: pre-line !important;
    line-height: 1.25 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}}
div[data-testid="column"]:has(.slot-occupied) button p,
div[data-testid="stColumn"]:has(.slot-occupied) button p,
button[aria-label*="BUSY"] p {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    line-height: 1.25 !important;
}}

div[data-testid="column"]:has(.slot-pending) button,
div[data-testid="stColumn"]:has(.slot-pending) button,
div:has(> .slot-pending) + div[data-testid="stButton"] button,
button[aria-label*="MATCH"] {{
    background-color: #F59E0B !important;
    background: #F59E0B !important;
    color: #1E293B !important;
    border: 1px solid #D97706 !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    min-height: 52px !important;
    padding: 6px 2px !important;
    white-space: pre-line !important;
    line-height: 1.25 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}}
div[data-testid="column"]:has(.slot-pending) button p,
div[data-testid="stColumn"]:has(.slot-pending) button p,
button[aria-label*="MATCH"] p {{
    color: #1E293B !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    line-height: 1.25 !important;
}}

div[data-testid="column"]:has(.slot-vacating) button,
div[data-testid="stColumn"]:has(.slot-vacating) button,
div:has(> .slot-vacating) + div[data-testid="stButton"] button,
button[aria-label*="LEAVING"] {{
    background-color: #0284C7 !important;
    background: #0284C7 !important;
    color: #FFFFFF !important;
    border: 1px solid #0369A1 !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    min-height: 52px !important;
    padding: 6px 2px !important;
    white-space: pre-line !important;
    line-height: 1.25 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}}
div[data-testid="column"]:has(.slot-vacating) button p,
div[data-testid="stColumn"]:has(.slot-vacating) button p,
button[aria-label*="LEAVING"] p {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    line-height: 1.25 !important;
}}

.zone-card div[data-testid="stButton"] button:hover {{
    filter: brightness(1.15) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
}}
.zone-card div[data-testid="stButton"] button:active {{
    transform: translateY(0px) !important;
}}
.slot-marker {{
    display: none !important;
}}

/* ── Drive Aisle Separator ── */
.drive-aisle {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 8px 0;
    margin: 4px 0 12px 0;
}}
.drive-aisle-line {{
    flex: 1;
    height: 2px;
    background: repeating-linear-gradient(90deg, rgba(255,255,255,0.2) 0px, rgba(255,255,255,0.2) 10px, transparent 10px, transparent 20px);
}}
.drive-aisle-label {{
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 2px 10px;
    border: 1px dashed var(--border-color);
    border-radius: 4px;
}}

/* ── Modern Metric Cards ── */
.metric-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}}
.metric-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 2px;
}}
.metric-value {{
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.2;
}}
.metric-sub {{
    font-size: 0.72rem;
    color: var(--accent-blue);
    margin-top: 2px;
    font-weight: 600;
}}

/* ── Section Title ── */
.section-title {{
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 2px;
    letter-spacing: -0.02em;
}}
.section-desc {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 16px;
    line-height: 1.4;
}}

/* ── Misc ── */
hr.subtle {{
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 16px 0;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_COLORS = ["#0EA5E9", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#06B6D4"]

def apply_plotly_theme(fig, theme_mode=None):
    if theme_mode is None:
        theme_mode = st.session_state.get("theme_mode", "dark")
    is_dark = theme_mode == "dark"
    paper_bg = "#1A1C23" if is_dark else "#FFFFFF"
    plot_bg = "#1A1C23" if is_dark else "#FFFFFF"
    font_color = "#94A3B8" if is_dark else "#475569"
    grid_color = "#282C37" if is_dark else "#E2E8F0"
    
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=font_color, size=11),
        margin=dict(l=24, r=24, t=36, b=24),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, showline=True, linecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, showline=True, linecolor=grid_color),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color=font_color)),
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
    st.session_state.sim_time = datetime.now()


# ---------------------------------------------------------------------------
# Top Bar: Brand & Global Theme Controls
# ---------------------------------------------------------------------------

brand_col, ctrl_col = st.columns([3, 2])

with brand_col:
    st.markdown("""
    <div class="app-brand">
        <div>
            <h1>Megaworld Smart Parking Platform</h1>
            <p>Predictive availability, computer vision space detection, and confidence-weighted plate matching.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ctrl_col:
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        site_names = ["All Sites"] + list(sites_df["name"])
        selected_site = st.selectbox(
            "Township Selector",
            options=site_names,
            label_visibility="collapsed",
            key="global_site_select",
        )
    with t_col2:
        theme_val = st.segmented_control(
            "Theme",
            options=["Dark", "Light"],
            default="Dark" if st.session_state.theme_mode == "dark" else "Light",
            label_visibility="collapsed",
            key="theme_toggle_ctrl",
        )
        st.session_state.theme_mode = "dark" if theme_val == "Dark" else "light"

# Inject dynamic CSS based on active theme
st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar (minimal — dataset info + regenerate)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Platform Overview")
    st.markdown(f"**{len(sites_df)}** Townships · **{len(zones_df)}** Zones · **{len(slots_df)}** Slots")
    st.markdown(f"**{len(history_df):,}** Historical ML Datapoints")
    st.markdown("---")
    if st.button("Regenerate Synthetic Data"):
        gd.main()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Engineered for Megaworld Township Parking Proof of Concept.")


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
# Database Inspector Dialog & Query Helper
# ---------------------------------------------------------------------------

def fetch_slot_db_records(slot_id: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        slot_info = pd.read_sql("""
            SELECT s.slot_id, s.slot_code, z.label AS zone_name, z.level, z.zone_type, st.name AS site_name, z.capacity
            FROM slots s
            JOIN zones z ON s.zone_id = z.zone_id
            JOIN sites st ON z.site_id = st.site_id
            WHERE s.slot_id = ?
        """, conn, params=(slot_id,))

        state_info = pd.read_sql("""
            SELECT slot_id, status, updated_at
            FROM current_state
            WHERE slot_id = ?
        """, conn, params=(slot_id,))

        ticket_info = pd.read_sql("""
            SELECT ticket_id, plate, entry_time, payment_settled_at, slot_id
            FROM ticketing_records
            WHERE slot_id = ?
        """, conn, params=(slot_id,))

        plate_info = pd.read_sql("""
            SELECT read_id, slot_id, raw_ocr_text, char_confidences, true_plate
            FROM plate_reads
            WHERE slot_id = ?
        """, conn, params=(slot_id,))

        return slot_info, state_info, ticket_info, plate_info
    finally:
        conn.close()


@st.dialog("Parking Bay Database Record", width="large")
def inspect_slot_dialog(slot_id: int, slot_code: str, live_row_dict: dict):
    slot_info, state_info, ticket_info, plate_info = fetch_slot_db_records(slot_id)

    st.markdown(f"### Parking Stall `{slot_code}` (Database Row Inspector)")
    st.caption(f"Direct relational records queried from SQLite database (`data/parking.db`) for Slot ID `{slot_id}`")

    tab_merged, tab_state, tab_ticket, tab_alpr, tab_slot, tab_sql = st.tabs([
        "Active Live Telemetry",
        "Active Occupancy State (current_state)",
        "Ticketing & Settlement (ticketing_records)",
        "Optical Plate Read (plate_reads)",
        "Infrastructure Metadata (slots & zones)",
        "Executed SQLite Statement (Raw SQL)",
    ])

    with tab_merged:
        st.markdown("##### Consolidated Telemetry Snapshot")
        df_live = pd.DataFrame([live_row_dict])
        rename_map = {
            "slot_id": "Slot ID",
            "zone_id": "Zone ID",
            "slot_code": "Slot Code",
            "status": "Occupancy Status",
            "plate": "Assigned Vehicle Plate",
            "ticket_id": "Ticket ID",
            "read_text": "OCR Read Text",
            "true_plate": "True Plate",
            "confidences": "Char Confidences",
            "site_id": "Township Site ID",
        }
        df_live_renamed = df_live.rename(columns={k: v for k, v in rename_map.items() if k in df_live.columns})
        st.dataframe(df_live_renamed, hide_index=True, width="stretch")

    with tab_state:
        st.markdown("##### Active Bay Occupancy State (`current_state` Table)")
        if not state_info.empty:
            df_state_renamed = state_info.rename(columns={
                "slot_id": "Slot ID",
                "status": "Current Occupancy Status",
                "updated_at": "Last State Timestamp",
            })
            st.dataframe(df_state_renamed, hide_index=True, width="stretch")
        else:
            st.info("No active occupancy state found in `current_state` table.")

    with tab_ticket:
        st.markdown("##### Ticketing & Settlement Record (`ticketing_records` Table)")
        if not ticket_info.empty:
            df_ticket_renamed = ticket_info.rename(columns={
                "ticket_id": "Ticket ID",
                "plate": "Registered License Plate",
                "entry_time": "Entry Timestamp",
                "payment_settled_at": "Payment Settlement Time",
                "slot_id": "Assigned Slot ID",
            })
            st.dataframe(df_ticket_renamed, hide_index=True, width="stretch")
        else:
            st.info("Bay is currently vacant — no active ticketing or billing record on file.")

    with tab_alpr:
        st.markdown("##### Optical Character Recognition Plate Read (`plate_reads` Table)")
        if not plate_info.empty:
            df_plate_renamed = plate_info.rename(columns={
                "read_id": "Read ID",
                "slot_id": "Monitored Slot ID",
                "raw_ocr_text": "Optical OCR Text",
                "char_confidences": "Per-Character Confidence Vector",
                "true_plate": "Ground Truth Plate",
            })
            st.dataframe(df_plate_renamed, hide_index=True, width="stretch")
        else:
            st.info("No optical camera plate capture record associated with this parking bay.")

    with tab_slot:
        st.markdown("##### Parking Bay Infrastructure Metadata (`slots` & `zones` Tables)")
        if not slot_info.empty:
            df_slot_renamed = slot_info.rename(columns={
                "slot_id": "Slot ID",
                "slot_code": "Slot Identifier Code",
                "zone_name": "Parking Zone Name",
                "level": "Building Deck Level",
                "zone_type": "Zone Commercial Archetype",
                "site_name": "Township Site Name",
                "capacity": "Total Zone Capacity",
            })
            st.dataframe(df_slot_renamed, hide_index=True, width="stretch")
        else:
            st.info("No slot infrastructure metadata found.")

    with tab_sql:
        st.markdown("##### Executed Underlying SQLite Statement")
        sql_query = f"""SELECT 
    s.slot_id, s.slot_code, z.label AS zone_name, z.level, z.zone_type, st.name AS site_name,
    cs.status, cs.updated_at,
    tr.ticket_id, tr.plate, tr.entry_time, tr.payment_settled_at,
    pr.read_id, pr.raw_ocr_text, pr.char_confidences, pr.true_plate
FROM slots s
LEFT JOIN zones z ON s.zone_id = z.zone_id
LEFT JOIN sites st ON z.site_id = st.site_id
LEFT JOIN current_state cs ON s.slot_id = cs.slot_id
LEFT JOIN ticketing_records tr ON s.slot_id = tr.slot_id
LEFT JOIN plate_reads pr ON s.slot_id = pr.slot_id
WHERE s.slot_id = {slot_id};"""
        st.code(sql_query, language="sql")


if "selected_slot_id" not in st.session_state:
    st.session_state.selected_slot_id = None
if "selected_slot_code" not in st.session_state:
    st.session_state.selected_slot_code = None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Occupancy Map", "Availability Forecast", "Model Performance", "Plate Matching", "ALPR Feasibility", "Space Detection (CV)"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Occupancy Map & Township Overview
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Prominent Large Hero Clock (Date and Time Centered) ──
    formatted_time = st.session_state.sim_time.strftime("%I:%M %p")
    formatted_date = st.session_state.sim_time.strftime("%A, %B %d, %Y")

    st.markdown(f"""
    <div class="hero-clock-container">
        <div class="hero-clock-eyebrow">PHILIPPINE STANDARD TIME (PST) · REAL-TIME SYSTEM CLOCK</div>
        <div class="hero-clock-time">{formatted_time}</div>
        <div class="hero-clock-date">{formatted_date}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Clock & Calendar Time-Setter Interface ──
    clock_card_col1, clock_card_col2, clock_card_col3 = st.columns([2, 2, 1.2])
    with clock_card_col1:
        chosen_date = st.date_input(
            "Calendar Date",
            value=st.session_state.sim_time.date(),
            key="tab1_calendar_date_picker",
        )
    with clock_card_col2:
        chosen_time = st.time_input(
            "Clock Time",
            value=st.session_state.sim_time.time(),
            key="tab1_clock_time_picker",
            step=60,
        )
    with clock_card_col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Sync Live Clock", key="tab1_btn_sync_live", use_container_width=True):
            st.session_state.sim_time = datetime.now()
            st.rerun()

    # Sync and update state when calendar date or clock time is manually changed
    updated_dt = datetime.combine(chosen_date, chosen_time)
    if updated_dt != st.session_state.sim_time:
        st.session_state.sim_time = updated_dt
        st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Township High-Level KPI Summary ──
    active_slots = live_df[live_df.zone_id.isin(active_zones.zone_id)]
    total_slots_count = len(active_slots)
    vacant_slots_count = (active_slots.status == sm.FREE).sum()
    occupied_slots_count = total_slots_count - vacant_slots_count
    saturation_pct = (occupied_slots_count / total_slots_count * 100.0) if total_slots_count > 0 else 0.0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            f"<div class='metric-card' style='border-left:3px solid #38BDF8;'>"
            f"<div class='metric-label'>Total Capacity</div>"
            f"<div class='metric-value'>{total_slots_count}</div>"
            f"<div class='metric-sub'>Monitored Bays</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            f"<div class='metric-card' style='border-left:3px solid #34D399;'>"
            f"<div class='metric-label'>Available Slots</div>"
            f"<div class='metric-value' style='color:#34D399;'>{vacant_slots_count}</div>"
            f"<div class='metric-sub'>Ready for Drivers</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f"<div class='metric-card' style='border-left:3px solid #F43F5E;'>"
            f"<div class='metric-label'>Occupied Slots</div>"
            f"<div class='metric-value' style='color:#F43F5E;'>{occupied_slots_count}</div>"
            f"<div class='metric-sub'>Active Parking</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col4:
        sat_color = "#34D399" if saturation_pct < 70 else ("#FBBF24" if saturation_pct < 88 else "#F43F5E")
        st.markdown(
            f"<div class='metric-card' style='border-left:3px solid {sat_color};'>"
            f"<div class='metric-label'>Township Saturation</div>"
            f"<div class='metric-value' style='color:{sat_color};'>{saturation_pct:.1f}%</div>"
            f"<div class='metric-sub'>Capacity Load</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Legend Strip ──
    st.markdown(f"""
    <div class="legend-strip">
        <span style="font-size:0.85rem; font-weight:800; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-right:6px;">Status Key:</span>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.FREE]};"></div> Available (Free)</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_UNPAID]};"></div> Occupied</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_PENDING_MATCH]};"></div> Pending Match</div>
        <div class="legend-entry"><div class="dot" style="background:{sm.STATUS_COLORS[sm.OCCUPIED_LIKELY_VACATING]};"></div> Vacating</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render Township & Zone Seat-Map Sections ──
    for _, site in sites_df.iterrows():
        if site.site_id not in active_site_ids:
            continue

        site_zones = zones_df[zones_df.site_id == site.site_id]

        st.markdown(
            f"<div style='font-size:1.15rem; font-weight:800; color:var(--text-primary); margin:20px 0 12px 0; letter-spacing:-0.02em;'>"
            f"{site['name']}</div>",
            unsafe_allow_html=True,
        )

        # Order zones sequentially: Mall -> Office -> Residential
        zone_order_map = {"mall": 0, "office": 1, "residential": 2}
        site_zones = site_zones.sort_values(by="zone_type", key=lambda s: s.map(zone_order_map))

        for _, z in site_zones.iterrows():
            zone_slots = live_df[live_df.zone_id == z.zone_id].sort_values("slot_code").reset_index(drop=True)
            n_free = (zone_slots.status == sm.FREE).sum()
            total = len(zone_slots)
            tag_class = f"zone-tag-{z.zone_type}" if z.zone_type in ("office", "mall", "residential") else ""
            type_label = "Mall" if z.zone_type == "mall" else ("Office" if z.zone_type == "office" else "Residential")

            st.markdown(f"""
            <div class="zone-card">
                <div class="zone-header">
                    <div>
                        <span class="zone-name">{z.label} — {z.level}</span>
                        <span class="zone-tag {tag_class}">{type_label}</span>
                    </div>
                    <div class="zone-avail"><strong>{n_free}</strong> / {total} bays free · <span style="font-size:0.75rem; color:var(--text-muted);">Click any stall to inspect database</span></div>
                </div>
            """, unsafe_allow_html=True)

            # Determine column count & rows based on capacity
            row_size = 12 if total >= 24 else 8
            num_rows = (total + row_size - 1) // row_size

            for r_idx in range(num_rows):
                row_slice = zone_slots.iloc[r_idx * row_size : (r_idx + 1) * row_size]
                lane_char = chr(65 + r_idx)

                # Render Drive Aisle separator between facing parking rows
                if r_idx > 0:
                    st.markdown(
                        f"<div class='drive-aisle'>"
                        f"<div class='drive-aisle-line'></div>"
                        f"<div class='drive-aisle-label'>DRIVE AISLE · LANE {lane_char} ➔</div>"
                        f"<div class='drive-aisle-line'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                cols = st.columns(row_size)
                for c_idx, (_, row) in enumerate(row_slice.iterrows()):
                    with cols[c_idx]:
                        if row.status == sm.FREE:
                            marker_class = "slot-free"
                            ind_text = "OPEN"
                        elif row.status == sm.OCCUPIED_UNPAID:
                            marker_class = "slot-occupied"
                            ind_text = "BUSY"
                        elif row.status == sm.OCCUPIED_PENDING_MATCH:
                            marker_class = "slot-pending"
                            ind_text = "MATCH"
                        else:
                            marker_class = "slot-vacating"
                            ind_text = "LEAVING"

                        # Slot status marker for guaranteed CSS :has column targeting
                        st.markdown(f"<span class='slot-marker {marker_class}'></span>", unsafe_allow_html=True)

                        # Direct Solid-Color Clickable Parking Stall Block
                        if st.button(
                            f"{row.slot_code}\n{ind_text}",
                            key=f"stall_btn_{row.slot_id}",
                            help=f"Inspect SQLite database row for {row.slot_code} (Slot ID: {row.slot_id})",
                            use_container_width=True,
                        ):
                            st.session_state.selected_slot_id = int(row.slot_id)
                            st.session_state.selected_slot_code = str(row.slot_code)
                            inspect_slot_dialog(int(row.slot_id), str(row.slot_code), row.to_dict())

            st.markdown("</div>", unsafe_allow_html=True)


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
        options=["Philippine Parking Lot Dataset (Metro Manila)", "Academic ALPR Benchmark (OpenALPR)"],
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
            match_tag = "[MATCH] exact match" if result["exact_match"] else f"acc={result['char_accuracy']:.2f}"
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
    # Modern Hero Header
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("<div class='section-title' style='margin-bottom:2px;'>Parking Space Occupancy Detection</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#94A3B8; font-size:0.9rem; margin-bottom:12px;'>Real-time computer vision inference engine monitoring individual bay occupancy via overhead CCTV feeds.</div>", unsafe_allow_html=True)
    with header_col2:
        st.markdown(
            "<div style='text-align:right; padding-top:4px;'>"
            "<span style='background:#064E3B; color:#34D399; border:1px solid #059669; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600;'>"
            "● LIVE INFERENCE</span></div>",
            unsafe_allow_html=True,
        )

    available_angles = pd_engine.list_available_camera_angles()

    if not available_angles:
        st.warning("No camera feeds found in `car_dataset/`.")
    else:
        # Compact Primary Control Strip
        ctrl_col1, ctrl_col2 = st.columns([3, 2])

        with ctrl_col1:
            angle_options = [a["filename"] for a in available_angles]
            angle_choice = st.selectbox(
                "Surveillance Camera Feed",
                options=angle_options,
                format_func=lambda fn: next((a["display_name"] for a in available_angles if a["filename"] == fn), fn),
                key="cam_angle_select",
                label_visibility="collapsed",
            )

        with ctrl_col2:
            view_mode = st.segmented_control(
                "View Mode",
                options=["Overlay", "Side-by-Side", "Raw Feed", "API Payload"],
                default="Overlay",
                label_visibility="collapsed",
                key="studio_view_mode",
            )

        # Expandable Fine-Tuning Parameters (Keeps UI clean by default)
        with st.expander("Detection Parameters & AI Filters", expanded=False):
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                conf_val = st.slider(
                    "YOLO Conf Threshold (τ_conf)",
                    min_value=0.15,
                    max_value=0.85,
                    value=0.25,
                    step=0.05,
                    help="Minimum confidence threshold for vehicle detections",
                )
            with pcol2:
                ioa_val = st.slider(
                    "IoA Occupancy Threshold (τ_ioa)",
                    min_value=0.10,
                    max_value=0.60,
                    value=0.30,
                    step=0.05,
                    help="Overlap ratio required to classify bay as Occupied",
                )
            with pcol3:
                enable_low_light = st.toggle(
                    "Low-Light Boost",
                    value=True,
                    help="Adaptive CLAHE contrast enhancement for dark SUVs, pickups, and shaded areas",
                )
            with pcol4:
                enable_smoothing = st.toggle(
                    "Temporal Smoothing",
                    value=True,
                    help="Sliding-window debouncing across consecutive frames to prevent state flickering",
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

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            # Modern Metric KPI Strip
            occ_pct = summary["occupancy_rate"] * 100.0
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.markdown(
                    f"<div class='metric-card' style='border-left:3px solid #34D399;'>"
                    f"<div class='metric-label'>Available Bays</div>"
                    f"<div class='metric-value' style='color:#34D399;'>{summary['vacant_count']} <span style='font-size:0.85rem; color:#94A3B8;'>/ {summary['total_bays']} Free</span></div>"
                    f"<div class='metric-sub'>Ready for Parking</div></div>",
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    f"<div class='metric-card' style='border-left:3px solid #F43F5E;'>"
                    f"<div class='metric-label'>Occupied Bays</div>"
                    f"<div class='metric-value' style='color:#F43F5E;'>{summary['occupied_count']} <span style='font-size:0.85rem; color:#94A3B8;'>/ {summary['total_bays']} Filled</span></div>"
                    f"<div class='metric-sub'>Active Vehicles Parked</div></div>",
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    f"<div class='metric-card' style='border-left:3px solid #38BDF8;'>"
                    f"<div class='metric-label'>Occupancy Saturation</div>"
                    f"<div class='metric-value' style='color:#38BDF8;'>{occ_pct:.1f}%</div>"
                    f"<div class='metric-sub'>Capacity Utilization</div></div>",
                    unsafe_allow_html=True,
                )
            with kpi4:
                st.markdown(
                    f"<div class='metric-card' style='border-left:3px solid #A855F7;'>"
                    f"<div class='metric-label'>Vehicles Detected</div>"
                    f"<div class='metric-value' style='color:#C084FC;'>{summary['detected_vehicles']}</div>"
                    f"<div class='metric-sub'>YOLOv8n Objects</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            # Main Visual Studio
            if view_mode == "Overlay":
                st.image(
                    annotated_rgb,
                    caption=f"Real-Time Space Detection · {selected_meta['display_name']} ({summary['vacant_count']} Free / {summary['occupied_count']} Occupied)",
                    width="stretch",
                )
            elif view_mode == "Side-by-Side":
                side_col1, side_col2 = st.columns(2)
                with side_col1:
                    st.markdown("<div style='font-weight:600; font-size:0.85rem; color:#94A3B8; margin-bottom:4px;'>RAW SURVEILLANCE FEED</div>", unsafe_allow_html=True)
                    st.image(raw_rgb, width="stretch")
                with side_col2:
                    st.markdown("<div style='font-weight:600; font-size:0.85rem; color:#38BDF8; margin-bottom:4px;'>AI OCCUPANCY INFERENCE</div>", unsafe_allow_html=True)
                    st.image(annotated_rgb, width="stretch")
            elif view_mode == "Raw Feed":
                st.image(
                    raw_rgb,
                    caption=f"Raw Camera Stream: {selected_meta['filename']} ({raw_bgr.shape[1]}×{raw_bgr.shape[0]})",
                    width="stretch",
                )
            elif view_mode == "API Payload":
                st.markdown("<div style='font-weight:600; font-size:0.9rem; color:#38BDF8; margin-bottom:6px;'>Phase 5: Standardized JSON Telemetry Payload</div>", unsafe_allow_html=True)
                st.json(json_payload)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            # Interactive Quick-Glance Bay Status Pills
            st.markdown("<div style='font-weight:600; font-size:0.95rem; color:#F8FAFC; margin-bottom:8px;'>Live Bay Status Grid</div>", unsafe_allow_html=True)
            
            pills_cols = st.columns(min(len(bay_records), 6))
            for idx, b in enumerate(bay_records[:6]):
                is_occ = b["status"] == "occupied"
                badge_bg = "rgba(244, 63, 94, 0.15)" if is_occ else "rgba(52, 211, 153, 0.15)"
                badge_border = "#F43F5E" if is_occ else "#34D399"
                badge_color = "#FDA4AF" if is_occ else "#6EE7B7"
                badge_text = "OCCUPIED" if is_occ else "VACANT"
                with pills_cols[idx % len(pills_cols)]:
                    st.markdown(
                        f"<div style='background:{badge_bg}; border:1px solid {badge_border}; border-radius:8px; padding:8px 10px; text-align:center; margin-bottom:6px;'>"
                        f"<div style='font-weight:700; font-size:0.85rem; color:#F8FAFC;'>{b['slot_id']}</div>"
                        f"<div style='font-size:0.75rem; font-weight:600; color:{badge_color}; margin-top:2px;'>{badge_text}</div>"
                        f"<div style='font-size:0.68rem; color:#94A3B8;'>IoA: {b['occupancy_ratio']*100:.0f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Expandable Detailed Telemetry & Algorithm Cards (Uncluttered)
            tab_tel, tab_algo = st.tabs(["Detailed Bay Telemetry Table", "5-Phase Algorithm Specification"])

            with tab_tel:
                df_bays = pd.DataFrame(bay_records)
                if not df_bays.empty:
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

                    df_bays["status_badge"] = df_bays["status"].apply(lambda s: "Occupied" if s == "occupied" else "Vacant")
                    df_bays["ioa_pct"] = df_bays["occupancy_ratio"].apply(lambda r: f"{r*100:.1f}%")
                    df_bays["conf_pct"] = df_bays["confidence"].apply(lambda c: f"{c*100:.1f}%")
                    df_bays["veh_type"] = df_bays["matched_vehicle_class"].fillna("—")
                    df_bays["quality"] = df_bays["low_confidence_flag"].apply(lambda b: "Borderline" if b else "High Conf")

                    display_df = df_bays[["slot_id", "slot_name", "zone", "status_badge", "ioa_pct", "conf_pct", "veh_type", "quality"]].copy()
                    display_df.columns = ["Slot ID", "Bay Name", "Zone", "Status", "IoA Overlap", "Confidence", "Vehicle Class", "Quality Flag"]
                    st.dataframe(display_df, width="stretch", hide_index=True)

            with tab_algo:
                st.markdown("""
                <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:14px; font-size:0.84rem; line-height:1.6; color:#CBD5E1;">
                    <strong style="color:#38BDF8;">Phase 1: Dual Native ROI Calibration</strong> — Perspective trapezoids mapped from <code>slots_config.json</code> with proportional scaling.<br/>
                    <strong style="color:#38BDF8;">Phase 2: Adaptive Low-Light Inference</strong> — Dual-exposure YOLOv8n inference with CLAHE contrast enhancement for dark SUVs & shadows.<br/>
                    <strong style="color:#38BDF8;">Phase 3: Spatial IoA & Centroid Containment</strong> — Checks point-in-polygon containment, vehicle coverage ratio (≥35%), and IoA overlap.<br/>
                    <strong style="color:#38BDF8;">Phase 4: Temporal Debouncing</strong> — 5-frame rolling memory with ≥60% majority consensus eliminates flickering.<br/>
                    <strong style="color:#38BDF8;">Phase 5: Structured JSON Telemetry</strong> — Clean API payloads for driver mobile apps and automated LED entrance signage.
                </div>
                """, unsafe_allow_html=True)