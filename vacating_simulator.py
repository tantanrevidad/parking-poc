"""
vacating_simulator.py — Real-Time Vacating Feature Simulation Engine
===================================================================
Powers the interactive 3-Way Split-Screen Vacating Simulator in Tab 1:
  Stage 1: Occupied — Unpaid (🔴 Vehicle Parked, Customer Shopping)
  Stage 2: Occupied — Likely Vacating Soon (🔵 Kiosk Payment Settled, 15-min Grace Period)
  Stage 3: Vehicle Egress in Progress (🚗 Camera / Sensor Detects Movement & Traversal)
  Stage 4: Available (🟢 Bay Released, Capacity +1, Digital Signage Updated)
"""

import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = Path(__file__).parent / "data" / "parking.db"

# 4-Stage Lifecycle State Definitions
STAGE_PARKED_UNPAID = "occupied_unpaid"
STAGE_PAYMENT_GRACE = "occupied_likely_vacating"
STAGE_DEPARTING = "occupied_departing"
STAGE_RELEASED = "available"

STAGE_META = {
    STAGE_PARKED_UNPAID: {
        "step_num": 1,
        "title": "Stage 1: Vehicle Parked (Unpaid)",
        "badge": "🔴 OCCUPIED — UNPAID",
        "color": "#EF4444",
        "bg_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "#DC2626",
        "description": "Vehicle is securely parked. Customer is actively dining or shopping inside the Megaworld lifestyle mall.",
        "kiosk_status": "Awaiting Payment Settlement",
        "signage_action": "Overhead LED: RED · Slot Marked Busy",
    },
    STAGE_PAYMENT_GRACE: {
        "step_num": 2,
        "title": "Stage 2: Payment Confirmed (Grace Period)",
        "badge": "🔵 LIKELY VACATING SOON",
        "color": "#38BDF8",
        "bg_color": "rgba(56, 189, 248, 0.18)",
        "border_color": "#0284C7",
        "description": "Customer settled parking fee at mall kiosk / GCash. 15-minute exit grace window activated.",
        "kiosk_status": "Payment Settled · Digital Receipt Issued",
        "signage_action": "Overhead LED: BLUE · Wayfinding App Notified",
    },
    STAGE_DEPARTING: {
        "step_num": 3,
        "title": "Stage 3: Vehicle Egress in Progress",
        "badge": "🟡 DEPARTURE DETECTED",
        "color": "#FBBF24",
        "bg_color": "rgba(245, 158, 11, 0.18)",
        "border_color": "#D97706",
        "description": "Driver started ignition and begun reversing. Computer Vision IoA overlap drops below 35%.",
        "kiosk_status": "Exit Barrier Approached",
        "signage_action": "Sensor: Motion Logged · Pre-allocation Armed",
    },
    STAGE_RELEASED: {
        "step_num": 4,
        "title": "Stage 4: Bay Released & Vacant",
        "badge": "🟢 AVAILABLE (FREE)",
        "color": "#34D399",
        "bg_color": "rgba(16, 185, 129, 0.18)",
        "border_color": "#059669",
        "description": "Vehicle cleared the parking stall. Bay marked available in SQLite database and available count incremented.",
        "kiosk_status": "Transaction Closed · Gate Barrier Opened",
        "signage_action": "Overhead LED: GREEN · Entrance Board (+1 Available)",
    },
}


def get_db_connection():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def calculate_parking_fee(park_duration_minutes: int = 135) -> Dict[str, Any]:
    """
    Computes Megaworld Commercial Township parking fee:
      - First 3 Hours (Flat Rate): ₱50.00
      - Succeeding Hours: ₱20.00 / hour
    """
    hours = park_duration_minutes / 60.0
    if hours <= 3.0:
        total_fee = 50.0
        breakdown = "First 3 Hours Flat Rate (₱50.00)"
    else:
        extra_hours = int(hours - 3.0) + (1 if (hours - 3.0) % 1 > 0 else 0)
        extra_fee = extra_hours * 20.0
        total_fee = 50.0 + extra_fee
        breakdown = f"First 3 Hours (₱50) + {extra_hours} Extra Hrs (₱{extra_fee:.0f})"

    return {
        "duration_str": f"{int(hours)}h {int(park_duration_minutes % 60)}m",
        "duration_minutes": park_duration_minutes,
        "total_fee": total_fee,
        "total_fee_str": f"₱{total_fee:.2f}",
        "breakdown": breakdown,
        "flat_rate": 50.0,
        "hourly_rate": 20.0,
    }


def fetch_candidate_simulation_slots() -> List[Dict[str, Any]]:
    """
    Queries candidate occupied slots available for simulation across townships.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT 
                s.slot_id,
                s.slot_code,
                z.label AS zone_name,
                z.zone_type,
                st.name AS site_name,
                cs.status,
                cs.updated_at,
                tr.ticket_id,
                tr.plate,
                tr.entry_time,
                tr.payment_settled_at
            FROM slots s
            JOIN zones z ON s.zone_id = z.zone_id
            JOIN sites st ON z.site_id = st.site_id
            JOIN current_state cs ON s.slot_id = cs.slot_id
            LEFT JOIN ticketing_records tr ON s.slot_id = tr.slot_id AND tr.payment_settled_at IS NULL
            ORDER BY s.slot_id ASC
        """
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def execute_stage_transition(
    slot_id: int,
    target_stage: str,
    ticket_id: Optional[str] = None,
    plate: Optional[str] = None,
    payment_method: str = "GCash",
) -> Dict[str, Any]:
    """
    Applies the state transition in data/parking.db and returns structured telemetry.
    """
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        if target_stage == STAGE_PARKED_UNPAID:
            db_status = "occupied_unpaid"
            cursor.execute("UPDATE current_state SET status = ?, updated_at = ? WHERE slot_id = ?", (db_status, now_iso, slot_id))
            
            plate_val = plate or "MAT2357"
            ticket_val = ticket_id or f"SIM-TKT-{slot_id:03d}"
            cursor.execute("DELETE FROM ticketing_records WHERE slot_id = ?", (slot_id,))
            entry_time = (datetime.datetime.now() - datetime.timedelta(hours=2, minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO ticketing_records (ticket_id, plate, entry_time, payment_settled_at, slot_id)
                VALUES (?, ?, ?, NULL, ?)
            """, (ticket_val, plate_val, entry_time, slot_id))
            conn.commit()

        elif target_stage == STAGE_PAYMENT_GRACE:
            db_status = "occupied_likely_vacating"
            cursor.execute("UPDATE current_state SET status = ?, updated_at = ? WHERE slot_id = ?", (db_status, now_iso, slot_id))
            cursor.execute("UPDATE ticketing_records SET payment_settled_at = ? WHERE slot_id = ?", (now_iso, slot_id))
            conn.commit()

        elif target_stage == STAGE_DEPARTING:
            db_status = "occupied_likely_vacating"
            cursor.execute("UPDATE current_state SET updated_at = ? WHERE slot_id = ?", (now_iso, slot_id))
            conn.commit()

        elif target_stage == STAGE_RELEASED:
            db_status = "available"
            cursor.execute("UPDATE current_state SET status = ?, updated_at = ? WHERE slot_id = ?", (db_status, now_iso, slot_id))
            conn.commit()

        telemetry_event = {
            "timestamp": now_iso,
            "event_type": f"PARKING_LIFECYCLE_{target_stage.upper()}",
            "slot_id": slot_id,
            "stage_key": target_stage,
            "status_label": STAGE_META[target_stage]["badge"],
            "payment_method": payment_method if target_stage == STAGE_PAYMENT_GRACE else None,
            "grace_period_seconds": 900 if target_stage == STAGE_PAYMENT_GRACE else 0,
            "api_payload": {
                "system": "MEGAWORLD_SMART_PARKING_V2",
                "slot_id": slot_id,
                "current_status": target_stage,
                "led_indicator": STAGE_META[target_stage]["color"],
                "downstream_signage_refresh": True,
            }
        }
        return telemetry_event

    finally:
        conn.close()


def generate_simulated_receipt(
    plate: str,
    ticket_id: str,
    duration_str: str,
    fee_str: str,
    payment_method: str = "GCash",
) -> str:
    """Generates an authentic Megaworld Mall POS digital receipt."""
    now_str = datetime.datetime.now().strftime("%b %d, %Y - %I:%M %p")
    return f"""
┌───────────────────────────────────────────────────┐
│        MEGAWORLD LIFESTYLE MALLS PARKING          │
│            OFFICIAL DIGITAL E-RECEIPT             │
├───────────────────────────────────────────────────┤
│ Transaction Date:   {now_str}        │
│ Ticket Identifier:  {ticket_id:<30}│
│ Registered Plate:   {plate:<30}│
│ Duration of Stay:   {duration_str:<30}│
│ Payment Method:     {payment_method:<30}│
│ Total Amount Paid:  {fee_str:<30}│
├───────────────────────────────────────────────────┤
│ GRACE PERIOD: 15 MINUTES TO CLEAR EXIT BARRIER    │
│ THANK YOU FOR VISITING MEGAWORLD TOWNSHIPS!       │
└───────────────────────────────────────────────────┘
"""
