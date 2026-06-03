"""
telemetry_generator.py
======================
Simulates realistic IoT time-series telemetry data for a 500-vehicle EV fleet.

Each vehicle record captures:
  - Battery voltage, current, temperature (cell & ambient)
  - State-of-Charge (SoC) and State-of-Health (SoH)
  - Cycle count, odometer, charging events
  - GPS coordinates (charging station proximity)
  - Derived thermal risk labels

Output: Injects all records into SQLite via db_manager.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sqlite3
import os
import sys

from db_manager import DatabaseManager

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Fleet Configuration ─────────────────────────────────────────────────────────
NUM_VEHICLES       = 500
RECORDS_PER_VEHICLE = 100          # ~100 time-steps per vehicle → 50 000 total rows
SIMULATION_DAYS    = 365           # Spread records across 1 operational year
START_DATE         = datetime(2025, 1, 1)

# EV Battery specs (NMC chemistry baseline)
NOMINAL_VOLTAGE    = 400.0         # V (pack level)
VOLTAGE_STD        = 12.0
NOMINAL_CAPACITY   = 75.0          # kWh
MAX_TEMP_SAFE      = 45.0          # °C
CRITICAL_TEMP      = 60.0          # °C

# Charging station grid (lat/lon clusters – US West)
STATION_CLUSTERS = [
    {"name": "LA Downtown",      "lat": 34.052, "lon": -118.244, "capacity": 50},
    {"name": "San Francisco",    "lat": 37.774, "lon": -122.419, "capacity": 40},
    {"name": "Seattle Hub",      "lat": 47.606, "lon": -122.332, "capacity": 35},
    {"name": "Phoenix Grid",     "lat": 33.449, "lon": -112.074, "capacity": 45},
    {"name": "Las Vegas Node",   "lat": 36.175, "lon": -115.137, "capacity": 30},
    {"name": "Portland Depot",   "lat": 45.523, "lon": -122.676, "capacity": 25},
    {"name": "San Diego South",  "lat": 32.715, "lon": -117.157, "capacity": 20},
    {"name": "Denver Central",   "lat": 39.739, "lon": -104.984, "capacity": 28},
]

# Vehicle model types with different degradation profiles
VEHICLE_MODELS = {
    "Tesla Model 3 LR":   {"deg_rate": 0.015, "cap_kwh": 82,  "range_km": 560},
    "Rivian R1T":         {"deg_rate": 0.020, "cap_kwh": 135, "range_km": 505},
    "Chevy Bolt EV":      {"deg_rate": 0.025, "cap_kwh": 65,  "range_km": 417},
    "Ford F-150 Lightning":{"deg_rate": 0.018,"cap_kwh": 131, "range_km": 483},
    "Hyundai Ioniq 6":    {"deg_rate": 0.012, "cap_kwh": 77,  "range_km": 614},
}


# ── Helper Functions ────────────────────────────────────────────────────────────

def assign_vehicle_metadata(vehicle_id: int) -> dict:
    """Generate static metadata for a vehicle."""
    model_name = random.choice(list(VEHICLE_MODELS.keys()))
    model      = VEHICLE_MODELS[model_name]
    station    = random.choice(STATION_CLUSTERS)
    manufacture_year = random.randint(2020, 2024)
    initial_soh = np.clip(np.random.normal(98.0, 1.5), 90, 100)

    return {
        "vehicle_id":       f"EV-{vehicle_id:04d}",
        "model":            model_name,
        "capacity_kwh":     model["cap_kwh"],
        "range_km":         model["range_km"],
        "deg_rate":         model["deg_rate"],
        "station_name":     station["name"],
        "station_lat":      station["lat"] + np.random.normal(0, 0.05),
        "station_lon":      station["lon"] + np.random.normal(0, 0.05),
        "manufacture_year": manufacture_year,
        "initial_soh":      initial_soh,
        "total_cycles":     random.randint(0, 800),
    }


def compute_soh(initial_soh: float, cycles: int, deg_rate: float,
                temperature_mean: float) -> float:
    """
    SoH degrades with cycle count and chronic high-temperature exposure.
    Model: SoH = initial - (deg_rate * cycles) - thermal_penalty
    """
    thermal_penalty = max(0, (temperature_mean - 35.0) * 0.008 * cycles / 100)
    soh = initial_soh - (deg_rate * cycles / 10.0) - thermal_penalty
    return float(np.clip(soh, 40.0, 100.0))


def thermal_risk_label(cell_temp: float, ambient_temp: float,
                        current: float, soc: float) -> str:
    """Rule-based thermal risk classification (ground truth for ML)."""
    risk_score = 0
    if cell_temp > 55:       risk_score += 3
    elif cell_temp > 45:     risk_score += 2
    elif cell_temp > 38:     risk_score += 1

    if ambient_temp > 40:    risk_score += 1
    if current > 150:        risk_score += 2
    elif current > 100:      risk_score += 1
    if soc > 95:             risk_score += 1
    if soc < 5:              risk_score += 1

    if risk_score >= 5:   return "CRITICAL"
    elif risk_score >= 3: return "HIGH"
    elif risk_score >= 1: return "MEDIUM"
    else:                 return "LOW"


def generate_vehicle_timeseries(meta: dict) -> list[dict]:
    """
    Generate RECORDS_PER_VEHICLE telemetry snapshots for one vehicle.
    Simulates realistic driving cycles: charging → discharging → idle.
    """
    records     = []
    cycles      = meta["total_cycles"]
    soc         = np.random.uniform(20, 95)       # starting SoC
    odometer    = np.random.uniform(0, 80_000)    # km
    time_offset = random.randint(0, SIMULATION_DAYS * 24 * 3600)
    ts          = START_DATE + timedelta(seconds=time_offset)

    # Season affects ambient temperature
    month = ts.month
    base_ambient = 15 + 12 * np.sin((month - 4) * np.pi / 6)  # seasonal curve

    for _ in range(RECORDS_PER_VEHICLE):
        # ── Simulate operational mode ──────────────────────────────────────
        mode = random.choices(
            ["driving", "charging", "idle"],
            weights=[0.55, 0.25, 0.20]
        )[0]

        ambient_temp = base_ambient + np.random.normal(0, 4.0)
        ambient_temp = np.clip(ambient_temp, -10, 50)

        if mode == "driving":
            current      = np.random.uniform(60, 200)     # A discharge
            voltage      = NOMINAL_VOLTAGE - (current * 0.05) + np.random.normal(0, 2)
            cell_temp    = ambient_temp + np.random.uniform(5, 20)
            soc         -= np.random.uniform(0.5, 2.5)
            odometer    += np.random.uniform(5, 40)
            distance_km  = np.random.uniform(5, 40)
            charging_eff = None

        elif mode == "charging":
            current      = -np.random.uniform(20, 120)    # negative = charging
            voltage      = NOMINAL_VOLTAGE + abs(current) * 0.02 + np.random.normal(0, 1)
            cell_temp    = ambient_temp + np.random.uniform(3, 12)
            soc_gain     = np.random.uniform(1, 8)
            charging_eff = np.random.uniform(88, 97)      # %
            soc         += soc_gain
            cycles      += soc_gain / 100.0               # partial cycle
            distance_km  = 0.0

        else:  # idle
            current      = np.random.uniform(-2, 2)       # parasitic draw
            voltage      = NOMINAL_VOLTAGE + np.random.normal(0, 0.5)
            cell_temp    = ambient_temp + np.random.uniform(0, 3)
            distance_km  = 0.0
            charging_eff = None

        # Clamp SoC
        soc = float(np.clip(soc, 2, 100))

        # ── Compute SoH ──────────────────────────────────────────────────
        soh = compute_soh(meta["initial_soh"], int(cycles), meta["deg_rate"], cell_temp)

        # ── Thermal risk ─────────────────────────────────────────────────
        risk = thermal_risk_label(cell_temp, ambient_temp, abs(current), soc)

        # ── Pack voltage noise ────────────────────────────────────────────
        voltage = float(np.clip(voltage, 300, 450))

        records.append({
            "vehicle_id":        meta["vehicle_id"],
            "model":             meta["model"],
            "timestamp":         ts.strftime("%Y-%m-%d %H:%M:%S"),
            "mode":              mode,
            "battery_voltage_v": round(voltage, 2),
            "battery_current_a": round(current, 2),
            "cell_temp_c":       round(cell_temp, 2),
            "ambient_temp_c":    round(ambient_temp, 2),
            "soc_pct":           round(soc, 2),
            "soh_pct":           round(soh, 2),
            "cycle_count":       round(cycles, 1),
            "odometer_km":       round(odometer, 1),
            "distance_km":       round(distance_km, 2),
            "charging_eff_pct":  round(charging_eff, 2) if charging_eff else None,
            "thermal_risk":      risk,
            "station_name":      meta["station_name"],
            "station_lat":       round(meta["station_lat"], 5),
            "station_lon":       round(meta["station_lon"], 5),
            "manufacture_year":  meta["manufacture_year"],
        })

        # Advance timestamp by 15–120 minutes
        ts += timedelta(minutes=random.randint(15, 120))

    return records


def generate_fleet_telemetry() -> pd.DataFrame:
    """
    Master function: generate telemetry for all NUM_VEHICLES vehicles.
    Returns a consolidated DataFrame.
    """
    print(f"[*] Generating telemetry for {NUM_VEHICLES} vehicles "
          f"({RECORDS_PER_VEHICLE} records each)...")

    all_records = []
    for vid in range(1, NUM_VEHICLES + 1):
        if vid % 50 == 0:
            print(f"   -> Vehicle {vid}/{NUM_VEHICLES} processed...")
        meta    = assign_vehicle_metadata(vid)
        records = generate_vehicle_timeseries(meta)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"[OK] Generated {len(df):,} total telemetry records.")
    return df


# ── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_fleet_telemetry()

    print("\n[*] Connecting to database and ingesting data...")
    db = DatabaseManager()
    db.create_tables()
    db.ingest_telemetry(df)
    db.close()

    print("\n[*] Sample records:")
    print(df.head(3).to_string(index=False))
    print(f"\n[DONE] Pipeline complete! {len(df):,} records stored in fleet_telemetry.db")
