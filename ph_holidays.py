"""
ph_holidays.py
--------------
Official Philippine National Holidays Calendar and helper functions.
Supports Regular Holidays, Special (Non-Working) Days, and movable holiday calculations
for 2024 through 2028.
"""

from datetime import date, datetime
from typing import Optional, Dict, Tuple

# Fixed-date Philippine National Holidays
# (month, day) -> (Holiday Name, Holiday Type)
FIXED_PH_HOLIDAYS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (1, 1): ("New Year's Day", "Regular Holiday"),
    (1, 2): ("Special Non-Working Day", "Special Non-Working Day"),
    (2, 25): ("EDSA People Power Revolution Anniversary", "Special Non-Working Day"),
    (4, 9): ("Araw ng Kagitingan (Day of Valor)", "Regular Holiday"),
    (5, 1): ("Labor Day", "Regular Holiday"),
    (6, 12): ("Independence Day", "Regular Holiday"),
    (8, 21): ("Ninoy Aquino Day", "Special Non-Working Day"),
    (11, 1): ("All Saints' Day", "Special Non-Working Day"),
    (11, 2): ("All Souls' Day", "Special Non-Working Day"),
    (11, 30): ("Bonifacio Day", "Regular Holiday"),
    (12, 8): ("Feast of the Immaculate Conception", "Special Non-Working Day"),
    (12, 24): ("Christmas Eve", "Special Non-Working Day"),
    (12, 25): ("Christmas Day", "Regular Holiday"),
    (12, 30): ("Rizal Day", "Regular Holiday"),
    (12, 31): ("Last Day of the Year (New Year's Eve)", "Special Non-Working Day"),
}

# Movable Philippine Holidays by Year (Easter-derived & Islamic lunar calendar dates)
# YYYY-MM-DD -> (Holiday Name, Holiday Type)
MOVABLE_PH_HOLIDAYS: Dict[str, Tuple[str, str]] = {
    # 2024
    "2024-02-10": ("Chinese New Year", "Special Non-Working Day"),
    "2024-03-28": ("Maundy Thursday", "Regular Holiday"),
    "2024-03-29": ("Good Friday", "Regular Holiday"),
    "2024-03-30": ("Black Saturday", "Special Non-Working Day"),
    "2024-04-10": ("Eidul Fitr (Feast of Ramadan)", "Regular Holiday"),
    "2024-06-17": ("Eidul Adha (Feast of Sacrifice)", "Regular Holiday"),
    "2024-08-26": ("National Heroes Day", "Regular Holiday"),
    # 2025
    "2025-01-29": ("Chinese New Year", "Special Non-Working Day"),
    "2025-03-31": ("Eidul Fitr (Feast of Ramadan)", "Regular Holiday"),
    "2025-04-17": ("Maundy Thursday", "Regular Holiday"),
    "2025-04-18": ("Good Friday", "Regular Holiday"),
    "2025-04-19": ("Black Saturday", "Special Non-Working Day"),
    "2025-06-07": ("Eidul Adha (Feast of Sacrifice)", "Regular Holiday"),
    "2025-08-25": ("National Heroes Day", "Regular Holiday"),
    # 2026
    "2026-02-17": ("Chinese New Year", "Special Non-Working Day"),
    "2026-03-20": ("Eidul Fitr (Feast of Ramadan)", "Regular Holiday"),
    "2026-04-02": ("Maundy Thursday", "Regular Holiday"),
    "2026-04-03": ("Good Friday", "Regular Holiday"),
    "2026-04-04": ("Black Saturday", "Special Non-Working Day"),
    "2026-05-27": ("Eidul Adha (Feast of Sacrifice)", "Regular Holiday"),
    "2026-08-31": ("National Heroes Day", "Regular Holiday"),
    # 2027
    "2027-02-06": ("Chinese New Year", "Special Non-Working Day"),
    "2027-03-10": ("Eidul Fitr (Feast of Ramadan)", "Regular Holiday"),
    "2027-03-25": ("Maundy Thursday", "Regular Holiday"),
    "2027-03-26": ("Good Friday", "Regular Holiday"),
    "2027-03-27": ("Black Saturday", "Special Non-Working Day"),
    "2027-05-17": ("Eidul Adha (Feast of Sacrifice)", "Regular Holiday"),
    "2027-08-30": ("National Heroes Day", "Regular Holiday"),
    # 2028
    "2028-01-26": ("Chinese New Year", "Special Non-Working Day"),
    "2028-02-28": ("Eidul Fitr (Feast of Ramadan)", "Regular Holiday"),
    "2028-04-13": ("Maundy Thursday", "Regular Holiday"),
    "2028-04-14": ("Good Friday", "Regular Holiday"),
    "2028-04-15": ("Black Saturday", "Special Non-Working Day"),
    "2028-05-05": ("Eidul Adha (Feast of Sacrifice)", "Regular Holiday"),
    "2028-08-28": ("National Heroes Day", "Regular Holiday"),
}


def get_ph_holiday_info(target: date | datetime) -> Optional[Tuple[str, str]]:
    """
    Given a date or datetime, returns (holiday_name, holiday_type) if it falls on
    an official Philippine Holiday, or None otherwise.
    """
    if isinstance(target, datetime):
        d = target.date()
    else:
        d = target

    # Check movable holidays first (exact YYYY-MM-DD match)
    iso_key = d.isoformat()
    if iso_key in MOVABLE_PH_HOLIDAYS:
        return MOVABLE_PH_HOLIDAYS[iso_key]

    # Check fixed-date holidays (MM, DD match)
    fixed_key = (d.month, d.day)
    if fixed_key in FIXED_PH_HOLIDAYS:
        return FIXED_PH_HOLIDAYS[fixed_key]

    # Check National Heroes Day fallback (last Monday of August) if not already mapped
    if d.month == 8 and d.weekday() == 0 and (d.day + 7 > 31):
        return ("National Heroes Day", "Regular Holiday")

    return None


def is_ph_holiday(target: date | datetime) -> bool:
    """Returns True if the given date is an official Philippine Holiday."""
    return get_ph_holiday_info(target) is not None


def get_ph_holiday_name(target: date | datetime) -> Optional[str]:
    """Returns the name of the Philippine Holiday, or None."""
    info = get_ph_holiday_info(target)
    return info[0] if info else None


def get_holiday_occupancy_factor(target: date | datetime, zone_type: str = "mall") -> float:
    """
    Returns an occupancy modifier reflecting real-world Philippine holiday mobility patterns:
      - Mall zones: Surge by +35% to +45% due to leisure, dining, and cinema visits.
      - Office zones: Plummet by -75% to -85% as corporate/BPO staff are on holiday.
      - Residential zones: Steady elevated daytime occupancy (+15% to +25%) as residents stay home.
    """
    if not is_ph_holiday(target):
        return 1.0

    if zone_type == "mall":
        return 1.40
    elif zone_type == "office":
        return 0.20
    elif zone_type == "residential":
        return 1.25
    return 1.10
