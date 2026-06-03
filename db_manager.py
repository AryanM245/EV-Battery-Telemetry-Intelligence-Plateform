"""
db_manager.py
=============
SQLite pipeline controller for the EVIoT Predictive Maintenance Dataset.

Schema reflects the real Kaggle dataset columns (30 features) plus derived
fields added by data_loader.py.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "fleet_telemetry.db")

# ── DDL ─────────────────────────────────────────────────────────────────────────

CREATE_TELEMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS fleet_telemetry (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id              TEXT    NOT NULL,
    Timestamp               TEXT,
    SoC                     REAL,
    SoH                     REAL,
    Battery_Voltage         REAL,
    Battery_Current         REAL,
    Battery_Temperature     REAL,
    Charge_Cycles           REAL,
    Motor_Temperature       REAL,
    Motor_Vibration         REAL,
    Motor_Torque            REAL,
    Motor_RPM               REAL,
    Power_Consumption       REAL,
    Brake_Pad_Wear          REAL,
    Brake_Pressure          REAL,
    Reg_Brake_Efficiency    REAL,
    Tire_Pressure           REAL,
    Tire_Temperature        REAL,
    Suspension_Load         REAL,
    Ambient_Temperature     REAL,
    Ambient_Humidity        REAL,
    Load_Weight             REAL,
    Driving_Speed           REAL,
    Distance_Traveled       REAL,
    Idle_Time               REAL,
    Route_Roughness         REAL,
    RUL                     REAL,
    Failure_Probability     INTEGER,
    Maintenance_Type        INTEGER,
    Maintenance_Label       TEXT,
    TTF                     REAL,
    Component_Health_Score  REAL,
    thermal_risk            TEXT,
    carbon_offset_kg        REAL,
    charging_eff_pct        REAL
);
"""

CREATE_VEHICLE_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS vehicle_summary (
    vehicle_id              TEXT PRIMARY KEY,
    total_records           INTEGER,
    total_distance_km       REAL,
    avg_soh                 REAL,
    min_soh                 REAL,
    avg_soc                 REAL,
    max_charge_cycles       REAL,
    avg_charging_eff        REAL,
    avg_component_health    REAL,
    critical_events         INTEGER,
    failure_events          INTEGER,
    avg_rul                 REAL,
    last_updated            TEXT
);
"""

CREATE_STATION_DEMAND_TABLE = """
CREATE TABLE IF NOT EXISTS station_demand (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name        TEXT,
    station_lat         REAL,
    station_lon         REAL,
    charge_sessions     INTEGER,
    total_energy_kwh    REAL,
    last_updated        TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_vehicle_id ON fleet_telemetry (vehicle_id);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp  ON fleet_telemetry (Timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_thermal    ON fleet_telemetry (thermal_risk);",
    "CREATE INDEX IF NOT EXISTS idx_failure    ON fleet_telemetry (Failure_Probability);",
]

BATCH_SIZE = 5_000

# Kaggle columns to ingest (in exact order matching INSERT statement)
KAGGLE_COLS = [
    "vehicle_id", "Timestamp",
    "SoC", "SoH", "Battery_Voltage", "Battery_Current", "Battery_Temperature",
    "Charge_Cycles", "Motor_Temperature", "Motor_Vibration", "Motor_Torque",
    "Motor_RPM", "Power_Consumption", "Brake_Pad_Wear", "Brake_Pressure",
    "Reg_Brake_Efficiency", "Tire_Pressure", "Tire_Temperature",
    "Suspension_Load", "Ambient_Temperature", "Ambient_Humidity",
    "Load_Weight", "Driving_Speed", "Distance_Traveled", "Idle_Time",
    "Route_Roughness", "RUL", "Failure_Probability", "Maintenance_Type",
    "Maintenance_Label", "TTF", "Component_Health_Score",
    "thermal_risk", "carbon_offset_kg", "charging_eff_pct",
]


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=-64000;")
        print(f"[DB] Connected: {db_path}")

    def __enter__(self):  return self
    def __exit__(self, *a): self.close()

    # ── Schema ───────────────────────────────────────────────────────────────

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute(CREATE_TELEMETRY_TABLE)
        cur.execute(CREATE_VEHICLE_SUMMARY_TABLE)
        cur.execute(CREATE_STATION_DEMAND_TABLE)
        for idx in CREATE_INDEXES:
            cur.execute(idx)
        self.conn.commit()
        print("[OK] Schema ready.")

    def drop_and_recreate(self):
        self.conn.execute("DROP TABLE IF EXISTS fleet_telemetry;")
        self.conn.execute("DROP TABLE IF EXISTS vehicle_summary;")
        self.conn.execute("DROP TABLE IF EXISTS station_demand;")
        self.conn.commit()
        self.create_tables()
        print("[OK] Database wiped and re-created.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_kaggle_data(self, df: pd.DataFrame):
        """Batch-insert the enriched Kaggle DataFrame."""
        df = df.copy()
        df["Timestamp"] = df["Timestamp"].astype(str)

        # Ensure all expected columns exist
        for col in KAGGLE_COLS:
            if col not in df.columns:
                df[col] = None

        placeholders = ",".join(["?"] * len(KAGGLE_COLS))
        insert_sql   = f"INSERT INTO fleet_telemetry ({','.join(KAGGLE_COLS)}) VALUES ({placeholders})"

        rows   = [tuple(row) for row in df[KAGGLE_COLS].itertuples(index=False, name=None)]
        cursor = self.conn.cursor()
        total  = len(rows)

        for start in range(0, total, BATCH_SIZE):
            batch = rows[start: start + BATCH_SIZE]
            cursor.executemany(insert_sql, batch)
            self.conn.commit()
            print(f"   -> Inserted {min(start + BATCH_SIZE, total):,}/{total:,} rows...")

        print(f"[OK] Ingestion complete: {total:,} rows.")
        self._refresh_vehicle_summary()
        self._refresh_station_demand_synthetic()

    # ── Summary Refresh ───────────────────────────────────────────────────────

    def _refresh_vehicle_summary(self):
        self.conn.execute("DELETE FROM vehicle_summary;")
        self.conn.execute("""
        INSERT INTO vehicle_summary
        SELECT
            vehicle_id,
            COUNT(*)                                        AS total_records,
            SUM(Distance_Traveled)                          AS total_distance_km,
            AVG(SoH)                                        AS avg_soh,
            MIN(SoH)                                        AS min_soh,
            AVG(SoC)                                        AS avg_soc,
            MAX(Charge_Cycles)                              AS max_charge_cycles,
            AVG(charging_eff_pct)                           AS avg_charging_eff,
            AVG(Component_Health_Score)                     AS avg_component_health,
            SUM(CASE WHEN thermal_risk IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END)
                                                            AS critical_events,
            SUM(Failure_Probability)                        AS failure_events,
            AVG(RUL)                                        AS avg_rul,
            datetime('now')                                 AS last_updated
        FROM fleet_telemetry
        GROUP BY vehicle_id;
        """)
        self.conn.commit()
        print("[OK] vehicle_summary refreshed.")

    def _refresh_station_demand_synthetic(self):
        """
        The Kaggle dataset has no station GPS — synthesise 8 charging node
        aggregates mapped to the same US-West grid used previously.
        """
        self.conn.execute("DELETE FROM station_demand;")
        stations = [
            ("LA Downtown",     34.052, -118.244),
            ("San Francisco",   37.774, -122.419),
            ("Seattle Hub",     47.606, -122.332),
            ("Phoenix Grid",    33.449, -112.074),
            ("Las Vegas Node",  36.175, -115.137),
            ("Portland Depot",  45.523, -122.676),
            ("San Diego South", 32.715, -117.157),
            ("Denver Central",  39.739, -104.984),
        ]
        total_rows = self.conn.execute("SELECT COUNT(*) FROM fleet_telemetry").fetchone()[0]
        import random, math
        random.seed(42)
        for name, lat, lon in stations:
            sessions    = random.randint(800, 3500)
            energy      = round(sessions * random.uniform(18, 55), 1)
            self.conn.execute("""
            INSERT INTO station_demand (station_name, station_lat, station_lon,
                                        charge_sessions, total_energy_kwh, last_updated)
            VALUES (?,?,?,?,?,datetime('now'))
            """, (name, lat, lon, sessions, energy))
        self.conn.commit()
        print("[OK] station_demand refreshed (synthetic grid).")

    # ── Query Helpers ─────────────────────────────────────────────────────────

    def query_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_fleet_kpis(self) -> dict:
        row = self.conn.execute("""
        SELECT
            COUNT(DISTINCT vehicle_id)          AS fleet_size,
            SUM(Distance_Traveled)              AS total_distance_km,
            AVG(charging_eff_pct)               AS avg_charging_eff,
            SUM(Failure_Probability)            AS failure_events,
            AVG(SoH)                            AS avg_soh,
            MIN(SoH)                            AS min_soh,
            AVG(Component_Health_Score)         AS avg_health,
            AVG(RUL)                            AS avg_rul,
            SUM(carbon_offset_kg)               AS total_carbon_offset_kg
        FROM fleet_telemetry
        """).fetchone()

        total_km = row[1] or 0
        return {
            "fleet_size":           int(row[0] or 0),
            "total_distance_km":    round(total_km, 1),
            "avg_charging_eff":     round(row[2] or 0, 2),
            "failure_events":       int(row[3] or 0),
            "avg_soh":              round(row[4] or 0, 2),
            "min_soh":              round(row[5] or 0, 2),
            "avg_health":           round(row[6] or 0, 4),
            "avg_rul":              round(row[7] or 0, 1),
            "carbon_offset_t":      round((row[8] or 0) / 1000, 2),
        }

    def get_vehicle_list(self) -> list:
        rows = self.conn.execute(
            "SELECT DISTINCT vehicle_id FROM fleet_telemetry ORDER BY vehicle_id"
        ).fetchall()
        return [r[0] for r in rows]

    def get_vehicle_history(self, vehicle_id: str) -> pd.DataFrame:
        return self.query_df(
            "SELECT * FROM fleet_telemetry WHERE vehicle_id = ? ORDER BY Timestamp",
            params=(vehicle_id,)
        )

    def get_thermal_alerts(self, risk_levels: tuple = ("HIGH", "CRITICAL")) -> pd.DataFrame:
        placeholders = ",".join("?" * len(risk_levels))
        return self.query_df(
            f"""SELECT vehicle_id, Timestamp, Battery_Temperature, Motor_Temperature,
                       Ambient_Temperature, Battery_Current, SoC, thermal_risk,
                       Failure_Probability, Component_Health_Score
                FROM fleet_telemetry
                WHERE thermal_risk IN ({placeholders})
                ORDER BY Timestamp DESC
                LIMIT 1000""",
            params=risk_levels
        )

    def get_station_demand(self) -> pd.DataFrame:
        return self.query_df("SELECT * FROM station_demand ORDER BY charge_sessions DESC")

    def get_degradation_curves(self, sample_n: int = 20) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT vehicle_id, Charge_Cycles, SoH, Component_Health_Score, Timestamp
        FROM fleet_telemetry
        WHERE vehicle_id IN (
            SELECT DISTINCT vehicle_id FROM fleet_telemetry
            ORDER BY RANDOM() LIMIT {sample_n}
        )
        ORDER BY vehicle_id, Charge_Cycles
        """)

    def get_ml_training_data(self) -> pd.DataFrame:
        return self.query_df("""
        SELECT
            SoC, SoH, Battery_Voltage, Battery_Current, Battery_Temperature,
            Charge_Cycles, Motor_Temperature, Motor_Vibration, Motor_Torque,
            Motor_RPM, Power_Consumption, Brake_Pad_Wear, Ambient_Temperature,
            Driving_Speed, Distance_Traveled, Route_Roughness, Load_Weight,
            RUL, Failure_Probability, Component_Health_Score, thermal_risk
        FROM fleet_telemetry
        WHERE SoH IS NOT NULL
        """)

    def table_counts(self) -> dict:
        tables = ["fleet_telemetry", "vehicle_summary", "station_demand"]
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}

    def close(self):
        if self.conn:
            self.conn.close()
            print("[DB] Connection closed.")


if __name__ == "__main__":
    with DatabaseManager() as db:
        counts = db.table_counts()
        print("\nTable counts:")
        for t, c in counts.items():
            print(f"  {t:<25} {c:>10,}")
        kpis = db.get_fleet_kpis()
        print("\nKPIs:", kpis)
