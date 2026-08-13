"""
state_machine.py
-----------------
Explicit finite-state machine for slot status, matching the original design
doc's model:

    FREE --(vehicle detected)--> OCCUPIED_UNPAID
    OCCUPIED_UNPAID --(plate match resolved)--> OCCUPIED_LIKELY_VACATING
    OCCUPIED_LIKELY_VACATING --(vehicle leaves)--> FREE
    OCCUPIED_UNPAID --(vehicle leaves)--> FREE

STATUS_LABELS / STATUS_COLORS are shared with the Streamlit app so the UI
and the underlying logic never drift apart.
"""

FREE = "free"
OCCUPIED_UNPAID = "occupied_unpaid"
OCCUPIED_PENDING_MATCH = "occupied_pending_match"   # present + has a paid ticket, not yet matched by CV/matcher
OCCUPIED_LIKELY_VACATING = "occupied_likely_vacating"

STATUS_LABELS = {
    FREE: "Free",
    OCCUPIED_UNPAID: "Occupied — Unpaid",
    OCCUPIED_PENDING_MATCH: "Occupied — Payment Pending Match",
    OCCUPIED_LIKELY_VACATING: "Likely Vacating Soon",
}

STATUS_COLORS = {
    FREE: "#2ecc71",                 # green
    OCCUPIED_UNPAID: "#e74c3c",      # red
    OCCUPIED_PENDING_MATCH: "#e67e22",  # orange
    OCCUPIED_LIKELY_VACATING: "#f1c40f",  # amber/yellow
}

VALID_TRANSITIONS = {
    FREE: {OCCUPIED_UNPAID, OCCUPIED_PENDING_MATCH},
    OCCUPIED_UNPAID: {OCCUPIED_LIKELY_VACATING, FREE},
    OCCUPIED_PENDING_MATCH: {OCCUPIED_LIKELY_VACATING, FREE},
    OCCUPIED_LIKELY_VACATING: {FREE},
}


def transition(current_status, event):
    """
    event in {"vehicle_arrived", "vehicle_arrived_ticket_pending",
              "match_resolved", "vehicle_left"}
    Returns the new status, or the current status unchanged if the event
    doesn't apply from this state (defensive — avoids illegal jumps).
    """
    if event == "vehicle_left":
        return FREE if current_status != FREE else FREE

    if event == "vehicle_arrived" and current_status == FREE:
        return OCCUPIED_UNPAID

    if event == "vehicle_arrived_ticket_pending" and current_status == FREE:
        return OCCUPIED_PENDING_MATCH

    if event == "match_resolved" and current_status in (OCCUPIED_UNPAID, OCCUPIED_PENDING_MATCH):
        return OCCUPIED_LIKELY_VACATING

    return current_status  # no-op for events that don't apply
