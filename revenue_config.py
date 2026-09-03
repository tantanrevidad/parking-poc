"""
revenue_config.py
=================
Configuration module for Megaworld township parking rates, Philippine retail
benchmarks, and international smart parking industry figures.

Every number defined here traces directly back to verifiable empirical research
or official Megaworld Lifestyle Malls rate cards.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Data Provenance Framework: 4 Sourcing Tiers
# ---------------------------------------------------------------------------
PROVENANCE_TIERS: Dict[str, Dict[str, str]] = {
    "A": {
        "tier": "Tier A",
        "badge": "ACTUAL RATE",
        "color": "#10B981",
        "bg_color": "rgba(16, 185, 129, 0.15)",
        "border_color": "#059669",
        "description": "Real Megaworld parking rates researched and verified from official signage and advisory notices.",
    },
    "B": {
        "tier": "Tier B",
        "badge": "DERIVED",
        "color": "#38BDF8",
        "bg_color": "rgba(56, 189, 248, 0.15)",
        "border_color": "#0284C7",
        "description": "Computed mathematically from existing POC database records using physical rate formulas.",
    },
    "C": {
        "tier": "Tier C",
        "badge": "PH BENCHMARK",
        "color": "#F59E0B",
        "bg_color": "rgba(245, 158, 11, 0.15)",
        "border_color": "#D97706",
        "description": "Philippine retail and real-estate industry benchmarks (Colliers PH, Megaworld Annual Reports).",
    },
    "D": {
        "tier": "Tier D",
        "badge": "INDUSTRY",
        "color": "#A855F7",
        "bg_color": "rgba(168, 85, 247, 0.15)",
        "border_color": "#9333EA",
        "description": "International academic and municipal smart parking pilots (SFpark, HAH Parking, PreciseParkLink).",
    },
}

# ---------------------------------------------------------------------------
# Tier A: Verified Megaworld Township Parking Rate Structures
# Sourced from official Megaworld Lifestyle Malls parking signage & MoneyMax.ph
# ---------------------------------------------------------------------------
PARKING_RATES: Dict[str, Dict[str, Any]] = {
    "Uptown Bonifacio": {
        "mall": {
            "source": "Uptown Mall official parking signage (2024-2025)",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "4th_to_7th_hr": 15.0,
            "7th_hr_plus_am": 100.0,  # Entry 6:00 AM – 12:00 NN (discourages office parkers in mall slots)
            "7th_hr_plus_pm": 30.0,   # Entry 12:01 PM – 5:59 AM
            "grace_period_mins": 15,
            "overnight_surcharge": 200.0,  # Enter before 12MN, exit after 12NN next day
        },
        "office": {
            "source": "Uptown Bonifacio Corporate Towers parking policy",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 200.0,
        },
        "residential": {
            "source": "Uptown Bonifacio residential visitor allocation",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 250.0,
        },
    },
    "Eastwood City": {
        "mall": {
            "source": "Eastwood Mall / MoneyMax.ph (2024-2025)",
            "structure": "tiered",
            "first_3hr_flat": 60.0,
            "succeeding_hr_weekday": 20.0,
            "weekend_holiday_flat": 60.0,  # Flat rate all day on Sat/Sun/Statutory Holidays
            "grace_period_mins": 15,
            "overnight_surcharge": 150.0,
        },
        "office": {
            "source": "Eastwood City Cyberpark parking policy",
            "structure": "tiered",
            "first_3hr_flat": 60.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 150.0,
        },
        "residential": {
            "source": "Eastwood residential towers visitor rate",
            "structure": "tiered",
            "first_3hr_flat": 60.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 200.0,
        },
    },
    "McKinley Hill": {
        "mall": {
            "source": "Venice Grand Canal Mall parking management (2024-2025)",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 150.0,
        },
        "office": {
            "source": "McKinley Hill Cyberpark parking rate schedule",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 150.0,
        },
        "residential": {
            "source": "McKinley Hill residential visitor allocation",
            "structure": "tiered",
            "first_3hr_flat": 50.0,
            "succeeding_hr": 20.0,
            "grace_period_mins": 15,
            "overnight_surcharge": 200.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Tier C & D: Market & Industry Research Benchmarks
# ---------------------------------------------------------------------------
INDUSTRY_BENCHMARKS: Dict[str, Any] = {
    # Philippine Retail Benchmarks (Tier C)
    "avg_mall_spend_per_visit_php": {
        "low": 1000.0,
        "mid": 2000.0,
        "high": 3000.0,
        "source": "Philippine retail industry consumer surveys, Colliers PH (2024)",
        "provenance": "Tier C",
    },
    "megaworld_daily_foot_traffic": {
        "value": 297000,
        "source": "Megaworld Corporation Annual Financial Report (FY2025)",
        "provenance": "Tier C",
    },
    "megaworld_mall_leasing_revenue_php": {
        "value": 6.9e9,
        "label": "₱6.9B (+9% YoY)",
        "source": "Megaworld FY2025 Financial Statement Disclosures",
        "provenance": "Tier C",
    },
    "dwell_spend_elasticity": {
        "value": 1.3,
        "meaning": "1% increase in customer dwell time corresponds to approx. 1.3% increase in retail spend",
        "source": "International Council of Shopping Centers (ICSC) & PathIntelligence Retail Analytics",
        "provenance": "Tier C",
    },
    # Smart Parking Industry Benchmarks (Tier D)
    "dynamic_pricing_uplift_pct": {
        "conservative": 15.0,
        "moderate": 25.0,
        "aggressive": 40.0,
        "source": "HAH Parking dynamic pricing deployments; SFpark municipal pilot evaluation",
        "provenance": "Tier D",
    },
    "manual_leakage_rate_pct": {
        "low": 5.0,
        "mid": 10.0,
        "high": 15.0,
        "source": "Vert.ai, PreciseParkLink parking revenue leakage audits",
        "provenance": "Tier D",
    },
    "cruising_minutes_without_guidance": {
        "value": 16.0,
        "source": "Urban mobility studies (Donald Shoup: High Cost of Free Parking)",
        "provenance": "Tier D",
    },
    "cruising_minutes_with_smart_guidance": {
        "value": 2.5,
        "source": "Smart parking wayfinding operator benchmarks",
        "provenance": "Tier D",
    },
    "co2_saved_per_parking_trip_kg": {
        "value": 0.48,
        "source": "Environmental Protection Agency (EPA) gasoline vehicle emissions model",
        "provenance": "Tier D",
    },
}
