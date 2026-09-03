"""
leakage_detector.py
===================
Revenue leakage recovery and overstay detection engine.
Identifies unpaid overstays, unauthorized vehicles, and projects potential revenue
recapture against international smart parking benchmarks (Tier D: 5–15% leakage).
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from revenue_config import INDUSTRY_BENCHMARKS
from revenue_engine import compute_ticket_revenue, get_db_connection

DB_PATH = Path(__file__).parent / "data" / "parking.db"


def compute_overstay_leakage(
    threshold_hours: float = 3.0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Detects vehicles currently in 'occupied_unpaid' state exceeding threshold_hours.
    Computes total uncollected fees, list of overstaying vehicles, and potential monthly recovery.

    Data Provenance:
        - Timestamps: Tier B (ticketing_records & current_state in SQLite)
        - Tariffs: Tier A (official Megaworld rate schedules)
        - Benchmark Context: Tier D (industry standard 5-15% manual leakage rate)
    """
    conn = get_db_connection(db_path)
    
    # Get current simulation timestamp from latest current_state timestamp
    latest_ts_row = conn.execute("SELECT max(updated_at) FROM current_state").fetchone()
    ref_time = datetime.fromisoformat(latest_ts_row[0]) if latest_ts_row and latest_ts_row[0] else datetime.now()

    query = """
    SELECT 
        cs.slot_id,
        s.slot_code,
        z.label as zone_label,
        z.zone_type,
        si.name as site_name,
        cs.status,
        cs.updated_at,
        t.ticket_id,
        t.plate,
        t.entry_time,
        t.payment_settled_at
    FROM current_state cs
    JOIN slots s ON cs.slot_id = s.slot_id
    JOIN zones z ON s.zone_id = z.zone_id
    JOIN sites si ON z.site_id = si.site_id
    LEFT JOIN ticketing_records t ON cs.slot_id = t.slot_id
        AND t.entry_time = (
            SELECT MAX(t2.entry_time) 
            FROM ticketing_records t2 
            WHERE t2.slot_id = cs.slot_id
        )
    WHERE cs.status IN ('occupied_unpaid', 'occupied_pending_match')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    flagged_vehicles = []
    total_unpaid_revenue = 0.0

    for _, row in df.iterrows():
        entry_str = row["entry_time"] or row["updated_at"]
        if not entry_str:
            continue
        try:
            entry_time = datetime.fromisoformat(entry_str)
        except Exception:
            continue

        dwell_hours = max(0.0, (ref_time - entry_time).total_seconds() / 3600.0)
        
        # Check if dwell time exceeds overstay threshold and ticket is unpaid
        is_unpaid = pd.isna(row["payment_settled_at"]) or str(row["payment_settled_at"]).strip() in ("", "None", "nan", "NaN")
        if dwell_hours >= threshold_hours and is_unpaid:
            uncollected_fee = compute_ticket_revenue(
                entry_time=entry_time,
                exit_time=ref_time,
                site_name=row["site_name"],
                zone_type=row["zone_type"],
            )
            total_unpaid_revenue += uncollected_fee
            
            flagged_vehicles.append({
                "slot_code": row["slot_code"],
                "plate": row["plate"] or "UNKNOWN",
                "site_name": row["site_name"],
                "zone_label": row["zone_label"],
                "zone_type": row["zone_type"].capitalize(),
                "entry_time": entry_time.strftime("%I:%M %p"),
                "dwell_hours": round(dwell_hours, 1),
                "uncollected_fee": uncollected_fee,
                "status": "Unpaid Overstay",
            })

    # Benchmark comparisons
    # In manual/semi-automated facilities, 5–15% of revenue leaks from tailgating, unmonitored exits, and overstays
    projected_daily_loss = total_unpaid_revenue * 2.5  # Normalized for 24h turnover
    projected_monthly_recovery = projected_daily_loss * 30.0

    return {
        "num_overstayers": len(flagged_vehicles),
        "total_unpaid_now": round(total_unpaid_revenue, 2),
        "projected_monthly_recovery": round(projected_monthly_recovery, 2),
        "threshold_hours": threshold_hours,
        "vehicles": flagged_vehicles,
        "industry_leakage_pct": INDUSTRY_BENCHMARKS["manual_leakage_rate_pct"]["mid"],
        "provenance": "Tier A Tariffs × Tier B Real-Time State Timestamps vs. Tier D 5–15% Leakage Benchmark",
    }
