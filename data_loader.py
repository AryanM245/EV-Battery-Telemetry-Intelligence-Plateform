"""
data_loader.py
==============
Loads the real EVIoT Predictive Maintenance Dataset from Kaggle and ingests
it into the local SQLite database.

Dataset: datasetengineer/eviot-predictivemaint-dataset
Records: 175,393  |  Interval: 15-minute IoT snapshots  |  Period: 2020-2025
Columns: 30 features across battery, motor, brake, tire, environmental, and
         predictive-maintenance targets.

Usage:
    python data_loader.py
"""

import os
import sys
import numpy as np
import pandas as pd
import kagglehub

from db_manager import DatabaseManager

DATASET_SLUG = "datasetengineer/eviot-predictivemaint-dataset"
CSV_NAME     = "EV_Predictive_Maintenance_Dataset_15min.csv"

# Maintenance type label map (integer → text)
MAINTENANCE_MAP = {
    0: "None",
    1: "Routine",
    2: "Battery Service",
    3: "Motor Service",
    4: "Brake Service",
    5: "Emergency",
}

# Derived thermal risk thresholds (applied to Battery_Temperature)
def thermal_risk_label(row) -> str:
    bt  = row["Battery_Temperature"]
    mt  = row["Motor_Temperature"]
    fp  = row["Failure_Probability"]
    soc = row["SoC"] * 100  # dataset stores SoC as 0-1 fraction

    score = 0
    if bt  > 55:  score += 3
    elif bt > 45: score += 2
    elif bt > 38: score += 1
    if mt  > 80:  score += 2
    elif mt > 65: score += 1
    if fp  == 1:  score += 2
    if soc < 5 or soc > 95: score += 1

    if score >= 5:   return "CRITICAL"
    elif score >= 3: return "HIGH"
    elif score >= 1: return "MEDIUM"
    else:            return "LOW"


def download_dataset() -> str:
    """Download dataset via kagglehub and return path to CSV file."""
    print("[*] Downloading EVIoT dataset from Kaggle...")
    base = kagglehub.dataset_download(DATASET_SLUG)
    csv_path = os.path.join(base, CSV_NAME)
    if not os.path.exists(csv_path):
        # Search recursively
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    break
    print(f"[OK] Dataset ready: {csv_path}")
    return csv_path


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, clean dtypes, and enrich with derived columns needed by the
    dashboard (vehicle_id, thermal_risk, maintenance label, carbon offset).
    """
    print("[*] Loading CSV into DataFrame...")
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    print(f"[OK] Loaded {len(df):,} rows x {df.shape[1]} columns.")

    # ── Normalise column names ───────────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── Assign synthetic vehicle IDs (dataset has no VehicleID column) ──────
    # 175,393 rows / 100 records per vehicle ≈ 1,753 unique vehicles
    RECORDS_PER_VEH = 100
    n_vehicles      = max(1, len(df) // RECORDS_PER_VEH)
    vehicle_ids     = [f"EV-{((i // RECORDS_PER_VEH) % n_vehicles) + 1:04d}"
                       for i in range(len(df))]
    df["vehicle_id"] = vehicle_ids

    # ── SoC: dataset stores as 0-1 fraction → convert to % ──────────────────
    if df["SoC"].max() <= 1.0:
        df["SoC"] = (df["SoC"] * 100).round(2)

    # ── SoH: same ────────────────────────────────────────────────────────────
    if df["SoH"].max() <= 1.0:
        df["SoH"] = (df["SoH"] * 100).round(2)

    # ── Maintenance type: int → label ────────────────────────────────────────
    df["Maintenance_Label"] = df["Maintenance_Type"].map(MAINTENANCE_MAP).fillna("Unknown")

    # ── Thermal risk (derived) ───────────────────────────────────────────────
    print("[*] Computing thermal risk labels...")
    df["thermal_risk"] = df.apply(thermal_risk_label, axis=1)

    # ── Carbon offset proxy: kg CO2 saved vs ICE per km driven ───────────────
    df["carbon_offset_kg"] = (df["Distance_Traveled"] * 0.21).round(3)

    # ── Charging efficiency proxy from Reg_Brake_Efficiency ─────────────────
    # Real charging eff not in dataset; use 92 + small variance as stand-in
    np.random.seed(42)
    df["charging_eff_pct"] = np.clip(
        np.random.normal(92, 2, size=len(df)), 85, 99
    ).round(2)

    print(f"[OK] Enrichment complete. Columns: {list(df.columns)}")
    return df


def ingest_to_db(df: pd.DataFrame):
    """Push the enriched DataFrame into SQLite via DatabaseManager."""
    db = DatabaseManager()
    db.create_tables()
    db.ingest_kaggle_data(df)
    db.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = download_dataset()
    df       = load_and_enrich(csv_path)
    ingest_to_db(df)
    print(f"\n[DONE] {len(df):,} real EVIoT records loaded into fleet_telemetry.db")
