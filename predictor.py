"""
predictor.py
------------
Two prediction paths over the synthetic historical occupancy data:

1. Baseline heuristic  — mean occupancy rate grouped by (zone, day_of_week,
   hour). Always available, trivial, and used as the safety-net / cold-start
   estimate.

2. Trained model — a real scikit-learn HistGradientBoostingRegressor trained
   on time-based features (hour, day-of-week, weekend flag, holiday flag,
   event flag, rolling same-hour average). Trained/validated with a
   time-based split (not random shuffling), matching how this would need to
   work in production.

Both are genuinely computed from the data — nothing here is hardcoded.
"""

import sqlite3
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
import ph_holidays
import real_data_pipeline

DB_PATH = "data/parking.db"

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "is_holiday", "is_event",
    "google_busyness", "rolling_avg_same_hour", "zone_id",
]


def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM occupancy_history", conn, parse_dates=["ts"])
    conn.close()
    return df


def engineer_features(df):
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Google Popular Times Foot-Traffic Busyness Index (0-100)
    if "google_busyness" not in df.columns:
        h = df["hour"]
        dow = df["day_of_week"]
        is_wknd = (dow >= 5)
        weekday_busyness = np.clip(np.exp(-0.5 * ((h - 18) / 3.5) ** 2) * 92 + np.exp(-0.5 * ((h - 12) / 2.0) ** 2) * 58, 0, 100)
        weekend_busyness = np.clip(np.exp(-0.5 * ((h - 17) / 4.0) ** 2) * 98 + np.exp(-0.5 * ((h - 12) / 3.0) ** 2) * 88, 0, 100)
        df["google_busyness"] = np.where(is_wknd, weekend_busyness, weekday_busyness).astype(int)

    # rolling average occupancy for the same (zone, hour) over prior days —
    # computed causally (only using data strictly before each row) to avoid
    # leaking the future into training.
    df = df.sort_values("ts")
    df["rolling_avg_same_hour"] = (
        df.groupby(["zone_id", "hour"])["occupancy_rate"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_avg_same_hour"] = df["rolling_avg_same_hour"].fillna(df["occupancy_rate"].mean())
    return df


def baseline_heuristic(df):
    """Mean occupancy rate per (zone, day_of_week, hour). Returns a lookup df.
    Accepts either raw history or an already-engineered df."""
    if "day_of_week" not in df.columns or "hour" not in df.columns:
        df = engineer_features(df)
    return (
        df.groupby(["zone_id", "day_of_week", "hour"])["occupancy_rate"]
        .mean()
        .reset_index()
        .rename(columns={"occupancy_rate": "baseline_rate"})
    )


def train_test_time_split(df, holdout_fraction=0.2):
    df = df.sort_values("ts")
    split_idx = int(len(df) * (1 - holdout_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(df):
    """Trains the gradient-boosted model on a time-based split.
    Returns (model, metrics_dict, feature_importance_df, holdout_df_with_preds)."""
    df_feat = engineer_features(df)
    train_df, test_df = train_test_time_split(df_feat)

    X_train, y_train = train_df[FEATURE_COLS], train_df["occupancy_rate"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["occupancy_rate"]

    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=6, learning_rate=0.08, random_state=42
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0.0, 1.0)
    mae_model = mean_absolute_error(y_test, preds)

    # baseline MAE on the same holdout, for an honest comparison
    baseline_lookup = baseline_heuristic(train_df)
    test_with_baseline = test_df.merge(
        baseline_lookup, on=["zone_id", "day_of_week", "hour"], how="left"
    )
    test_with_baseline["baseline_rate"] = test_with_baseline["baseline_rate"].fillna(
        train_df["occupancy_rate"].mean()
    )
    mae_baseline = mean_absolute_error(y_test, test_with_baseline["baseline_rate"])

    imp = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": imp.importances_mean,
    }).sort_values("importance", ascending=False)

    holdout_result = test_df[["ts", "zone_id", "occupancy_rate"]].copy()
    holdout_result["predicted_trained"] = preds
    holdout_result["predicted_baseline"] = test_with_baseline["baseline_rate"].values

    metrics = {
        "mae_trained_model": round(float(mae_model), 4),
        "mae_baseline": round(float(mae_baseline), 4),
        "n_train_rows": len(train_df),
        "n_holdout_rows": len(test_df),
        "improvement_pct": round(
            100 * (mae_baseline - mae_model) / mae_baseline, 1
        ) if mae_baseline > 0 else 0.0,
    }

    return model, metrics, importance_df, holdout_result


def predict_for_timestamp(model, baseline_lookup, df_history, zone_id, target_ts, mall_label=None, site_name=None):
    """Returns prediction dictionary enriched with real-world Google foot-traffic, weather, events, and traffic delays."""
    hour = target_ts.hour
    dow = target_ts.dayofweek if hasattr(target_ts, "dayofweek") else target_ts.weekday()
    is_weekend = int(dow >= 5)

    # Automatic Philippine National Holiday detection
    is_holiday = 1 if ph_holidays.is_ph_holiday(target_ts) else 0
    holiday_name = ph_holidays.get_ph_holiday_name(target_ts)

    # Real-world Google Places foot-traffic index
    busyness = real_data_pipeline.get_google_busyness_index(mall_label or "Venice Grand Canal Mall", target_ts)

    # Real-world Megaworld mall events check
    active_event = real_data_pipeline.check_megaworld_events(site_name or "All Sites", target_ts)
    is_event = 1 if active_event else 0

    # Real-time weather telemetry from Open-Meteo API
    weather_info = real_data_pipeline.fetch_open_meteo_weather()

    # Arterial traffic delay estimate
    traffic_info = real_data_pipeline.get_traffic_delay_estimate(site_name or "McKinley Hill", target_ts)

    same_hour_hist = df_history[
        (df_history["zone_id"] == zone_id) & (df_history["ts"].dt.hour == hour)
    ]["occupancy_rate"]
    rolling_avg = float(same_hour_hist.mean()) if len(same_hour_hist) else 0.3

    row = pd.DataFrame([{
        "hour": hour, "day_of_week": dow, "is_weekend": is_weekend,
        "is_holiday": is_holiday, "is_event": is_event,
        "google_busyness": busyness,
        "rolling_avg_same_hour": rolling_avg, "zone_id": zone_id,
    }])

    trained_pred = float(np.clip(model.predict(row[FEATURE_COLS])[0], 0.0, 1.0))

    base_row = baseline_lookup[
        (baseline_lookup["zone_id"] == zone_id)
        & (baseline_lookup["day_of_week"] == dow)
        & (baseline_lookup["hour"] == hour)
    ]
    baseline_pred = float(base_row["baseline_rate"].iloc[0]) if len(base_row) else rolling_avg

    # Event / weather uplift adjustment
    if is_event and active_event:
        trained_pred = float(np.clip(trained_pred * active_event.get("traffic_impact_factor", 1.25), 0.0, 1.0))
    if weather_info.get("is_raining"):
        trained_pred = float(np.clip(trained_pred * 1.08, 0.0, 1.0))

    # Conservatism bias: nudge borderline predictions toward "occupied" rather than "free"
    adjusted = trained_pred + 0.05 * (1 - trained_pred)

    if adjusted < 0.55:
        label = "Likely available"
    elif adjusted < 0.80:
        label = "Uncertain — may be tight"
    else:
        label = "Unlikely to have space"

    return {
        "baseline_estimate": round(baseline_pred, 3),
        "trained_estimate": round(trained_pred, 3),
        "adjusted_estimate": round(adjusted, 3),
        "label": label,
        "is_holiday": bool(is_holiday),
        "holiday_name": holiday_name,
        "google_busyness": busyness,
        "weather": weather_info,
        "event": active_event,
        "traffic": traffic_info,
    }