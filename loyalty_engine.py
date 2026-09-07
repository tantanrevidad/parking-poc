"""
loyalty_engine.py
=================
Repeat-visitor recognition and customer loyalty spend analytics.
Complies with the Philippine Data Privacy Act of 2012 (RA 10173) by operating
exclusively on salted SHA-256 hashed plate identifiers — never storing or
exposing raw license plate strings in analytics or presentation layers.
"""

import hashlib
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from revenue_config import LOYALTY_BENCHMARKS, PROVENANCE_TIERS

DB_PATH = Path(__file__).parent / "data" / "parking.db"
DEFAULT_SALT = os.environ.get("PLATE_HASH_SALT", "megaworld_smart_parking_2026_salt")


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = db_path or DB_PATH
    return sqlite3.connect(str(target_path), check_same_thread=False)


def hashed_plate(plate: str, salt: Optional[str] = None) -> str:
    """
    Computes a cryptographic 16-character salted SHA-256 hash of a license plate.
    Prevents identification of vehicle owners under RA 10173 while enabling
    stable cohort aggregation across time windows.
    """
    effective_salt = salt or DEFAULT_SALT
    salted_string = f"{effective_salt}:{plate.strip().upper()}"
    return hashlib.sha256(salted_string.encode("utf-8")).hexdigest()[:16]


def _empty_loyalty_result(lookback_days: int = 28) -> dict:
    return {
        "lookback_days": lookback_days,
        "total_visitors": 0,
        "total_unique_visitors": 0,
        "total_tickets": 0,
        "total_tickets_analyzed": 0,
        "repeat_visitor_rate": 0.0,
        "repeat_visitor_rate_pct": 0.0,
        "returning_visitors": 0,
        "loyal_visitors": 0,
        "loyal_share_pct": 0.0,
        "incremental_spend_php": 0.0,
        "segment_counts": {"New": 0, "Returning": 0, "Loyal": 0},
        "segment_ticket_share": {"New": 0.0, "Returning": 0.0, "Loyal": 0.0},
        "archetype_breakdown": [],
        "provenance": "Tier B Derived Cohorts × Tier D (+67% Repeat Spend Premium)",
    }


def compute_repeat_visitor_segments(
    site_name: Optional[str] = None,
    lookback_days: int = 28,
    base_spend_php: float = 2000.0,
    salt: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyzes repeat-visitor cohorts over a given lookback period (7, 14, or 28 days).
    Segments visitors into:
        - New: 1 visit
        - Returning: 2–3 visits
        - Loyal: 4+ visits

    Data Provenance:
        - Visit counts: Tier B (ticketing_records with salted hashing)
        - Loyalty spend premium: Tier D (compiled retail loyalty analytics +67%)
        - Baseline spend: Tier C (Colliers PH retail surveys)
    """
    conn = get_db_connection(db_path)
    
    # Get latest entry time for stable reference cutoff
    latest_ts_row = conn.execute("SELECT max(entry_time) FROM ticketing_records").fetchone()
    if not latest_ts_row or not latest_ts_row[0]:
        conn.close()
        return _empty_loyalty_result(lookback_days)

    latest_dt = datetime.fromisoformat(latest_ts_row[0])
    cutoff_dt = latest_dt - timedelta(days=lookback_days)
    cutoff_str = cutoff_dt.isoformat()

    query = """
    SELECT 
        t.ticket_id,
        t.plate,
        t.entry_time,
        z.zone_type,
        si.name as site_name
    FROM ticketing_records t
    JOIN slots s ON t.slot_id = s.slot_id
    JOIN zones z ON s.zone_id = z.zone_id
    JOIN sites si ON z.site_id = si.site_id
    WHERE t.entry_time >= ?
    """
    params = [cutoff_str]
    if site_name and site_name != "All Townships":
        query += " AND si.name = ?"
        params.append(site_name)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return _empty_loyalty_result(lookback_days)

    # Hash all plates immediately — purge raw plate text from analysis pipeline
    effective_salt = salt or DEFAULT_SALT
    df["hashed_id"] = df["plate"].apply(lambda p: hashed_plate(p, effective_salt))
    df = df.drop(columns=["plate"])

    # Count visits per hashed vehicle ID over the window
    visitor_counts = df.groupby("hashed_id").size().to_dict()
    df["visit_frequency"] = df["hashed_id"].map(visitor_counts)

    def assign_segment(freq: int) -> str:
        if freq == 1:
            return "New"
        elif 2 <= freq <= 3:
            return "Returning"
        else:
            return "Loyal"

    df["segment"] = df["visit_frequency"].apply(assign_segment)

    # Unique visitor counts by segment
    unique_visitors_df = df.drop_duplicates(subset=["hashed_id"])
    seg_visitor_counts = unique_visitors_df["segment"].value_counts().to_dict()
    n_new_visitors = seg_visitor_counts.get("New", 0)
    n_ret_visitors = seg_visitor_counts.get("Returning", 0)
    n_loy_visitors = seg_visitor_counts.get("Loyal", 0)
    total_unique_visitors = len(unique_visitors_df)

    # Ticket volume share by segment
    total_tickets = len(df)
    ticket_counts = df["segment"].value_counts().to_dict()
    new_tickets = ticket_counts.get("New", 0)
    returning_tickets = ticket_counts.get("Returning", 0)
    loyal_tickets = ticket_counts.get("Loyal", 0)
    repeat_tickets = returning_tickets + loyal_tickets

    repeat_visitor_rate = (repeat_tickets / max(1, total_tickets)) * 100.0
    loyal_share_pct = (loyal_tickets / max(1, total_tickets)) * 100.0

    # Incremental spend calculation
    # Returning/Loyal visitors spend approx 67% more per visit (Tier D compiled retail loyalty research)
    repeat_premium = LOYALTY_BENCHMARKS["repeat_customer_spend_premium_pct"]["value"] / 100.0  # 0.67
    incremental_spend_php = repeat_tickets * base_spend_php * repeat_premium

    # Archetype breakdown (New vs. Returning vs. Loyal share per zone type)
    df["zone_archetype"] = df["zone_type"].apply(lambda z: z.capitalize())
    archetype_grouped = df.groupby(["zone_archetype", "segment"]).size().unstack(fill_value=0)
    
    archetype_data = []
    for arch in ["Mall", "Office", "Residential"]:
        if arch in archetype_grouped.index:
            row = archetype_grouped.loc[arch]
            tot_arch = max(1, row.sum())
            archetype_data.append({
                "zone_archetype": arch,
                "New": round((row.get("New", 0) / tot_arch) * 100.0, 1),
                "Returning": round((row.get("Returning", 0) / tot_arch) * 100.0, 1),
                "Loyal": round((row.get("Loyal", 0) / tot_arch) * 100.0, 1),
                "new_pct": round((row.get("New", 0) / tot_arch) * 100.0, 1),
                "returning_pct": round((row.get("Returning", 0) / tot_arch) * 100.0, 1),
                "loyal_pct": round((row.get("Loyal", 0) / tot_arch) * 100.0, 1),
                "total_visits": int(tot_arch),
            })

    return {
        "lookback_days": lookback_days,
        "total_visitors": total_unique_visitors,
        "total_unique_visitors": total_unique_visitors,
        "total_tickets": total_tickets,
        "total_tickets_analyzed": total_tickets,
        "repeat_visitor_rate": round(repeat_visitor_rate, 1),
        "repeat_visitor_rate_pct": round(repeat_visitor_rate, 1),
        "returning_visitors": n_ret_visitors,
        "loyal_visitors": n_loy_visitors,
        "loyal_share_pct": round(loyal_share_pct, 1),
        "incremental_spend_php": round(incremental_spend_php, 2),
        "segment_counts": {
            "New": n_new_visitors,
            "Returning": n_ret_visitors,
            "Loyal": n_loy_visitors,
        },
        "segment_ticket_share": {
            "New": round((new_tickets / max(1, total_tickets)) * 100.0, 1),
            "Returning": round((returning_tickets / max(1, total_tickets)) * 100.0, 1),
            "Loyal": round((loyal_tickets / max(1, total_tickets)) * 100.0, 1),
        },
        "archetype_breakdown": archetype_data,
        "provenance": "Tier B Hashed Cohorts × Tier C (₱1k–₱3k Spend) × Tier D (+67% Loyalty Premium)",
    }
