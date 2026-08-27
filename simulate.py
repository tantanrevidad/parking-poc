"""
simulate.py
-----------
Drives the "live" simulated state for an arbitrary point in time, reusing
the exact same daily-occupancy-curve / event / holiday logic as
generate_data.py (imported, not duplicated) so the historical data and the
live view stay internally consistent with each other.

For each simulated moment: decides which slots are occupied, generates a
plate + OCR read for each occupied slot, marks some as "payment settled",
then runs the REAL matcher (matcher.py) against a per-zone candidate pool
to resolve which occupied slots become "Likely Vacating Soon".
"""

import random
import sqlite3

import numpy as np
import pandas as pd

import generate_data as gd
import matcher
import state_machine as sm


def load_static_config():
    conn = sqlite3.connect(gd.DB_PATH)
    sites = pd.read_sql("SELECT * FROM sites", conn)
    zones = pd.read_sql("SELECT * FROM zones", conn)
    slots = pd.read_sql("SELECT * FROM slots", conn)
    holidays = pd.read_sql("SELECT * FROM holidays", conn).to_dict("records")
    events = pd.read_sql("SELECT * FROM events", conn).to_dict("records")
    conn.close()
    return sites, zones, slots, holidays, events


def simulate_current_state(sim_time, zones_df, slots_df, holidays, events, sites_df=None):
    """
    Returns a DataFrame, one row per slot, with columns:
        slot_id, zone_id, slot_code, status, plate, ticket_id,
        read_text, confidences, match_info (dict or None)
    """
    # Stable seed based on 5-minute simulation blocks so interactions don't reshuffle lot
    seed = int(sim_time.replace(second=0, microsecond=0).timestamp()) // 300
    rng = random.Random(seed)

    hour_frac = sim_time.hour + sim_time.minute / 60.0
    is_weekend = sim_time.weekday() >= 5

    rows = []

    for _, z in zones_df.iterrows():
        zone_slots = slots_df[slots_df.zone_id == z.zone_id]
        zone_type = z.zone_type if hasattr(z, "zone_type") else "mall"
        # Look up site name for event filtering
        site_name = None
        if sites_df is not None and hasattr(z, "site_id"):
            site_row = sites_df[sites_df.site_id == z.site_id]
            if len(site_row):
                site_name = site_row.iloc[0]["name"]
        base_rate = gd.daily_occupancy_curve(hour_frac, is_weekend, zone_type)
        mult = gd.event_multiplier(sim_time, events, site_name) * gd.holiday_multiplier(sim_time, holidays, zone_type)
        rate = float(np.clip(base_rate * mult, 0.0, 1.0))
        n_occupied = int(round(rate * len(zone_slots)))
        occupied_ids = set(rng.sample(list(zone_slots.slot_id), min(n_occupied, len(zone_slots))))

        zone_pool = []  # candidate tickets for this zone (paid, unmatched)
        pending_rows_idx = []

        for _, s in zone_slots.iterrows():
            sid = s.slot_id
            if sid not in occupied_ids:
                rows.append({
                    "slot_id": sid, "zone_id": z.zone_id, "slot_code": s.slot_code,
                    "status": sm.FREE, "plate": None, "ticket_id": None,
                    "read_text": None, "confidences": None, "match_info": None,
                })
                continue

            plate = gd.random_plate()
            has_paid = rng.random() < 0.45
            read_text, confs = gd.corrupt_plate(plate)
            ticket_id = f"SIM-{sid}-{seed % 10000}"

            row = {
                "slot_id": sid, "zone_id": z.zone_id, "slot_code": s.slot_code,
                "status": sm.OCCUPIED_PENDING_MATCH if has_paid else sm.OCCUPIED_UNPAID,
                "plate": plate, "ticket_id": ticket_id if has_paid else None,
                "read_text": read_text, "confidences": confs, "match_info": None,
            }
            rows.append(row)
            pending_rows_idx.append(len(rows) - 1)
            if has_paid:
                zone_pool.append({"ticket_id": ticket_id, "plate": plate})

        # Run the real matcher for every occupied slot in this zone against
        # the zone's candidate pool of paid-but-unresolved tickets.
        for idx in pending_rows_idx:
            row = rows[idx]
            if row["status"] not in (sm.OCCUPIED_PENDING_MATCH, sm.OCCUPIED_UNPAID):
                continue
            result = matcher.match_plate(row["read_text"], row["confidences"], zone_pool)
            row["match_info"] = result
            if result["resolved"] and result["matched_ticket_id"] == row["ticket_id"]:
                row["status"] = sm.OCCUPIED_LIKELY_VACATING

    return pd.DataFrame(rows)