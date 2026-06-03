"""
app.py
======
Executive Dashboard — EVIoT Predictive Maintenance Analytics
Real dataset: datasetengineer/eviot-predictivemaint-dataset (175,393 records)

Pages:
  1. Fleet Overview        — KPIs, health distribution, failure trend
  2. Degradation Analysis  — SoH/RUL curves, component health, drill-down
  3. Station Heatmap       — Geographic charging node demand
  4. Thermal Safety Log    — Overheating incidents, motor/battery alerts
  5. Predictive Analytics  — 4-model ML prediction panel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Fleet Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background: linear-gradient(135deg,#080c18 0%,#0d1b2a 60%,#080c18 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0a0f1e 0%,#141d2e 100%);
  border-right: 1px solid rgba(0,212,255,.15);
}

/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg,rgba(0,212,255,.08),rgba(0,100,200,.05));
  border: 1px solid rgba(0,212,255,.22);
  border-radius: 16px; padding: 18px 20px;
  text-align: center; transition: all .3s;
}
.kpi-card:hover {
  border-color: rgba(0,212,255,.55);
  box-shadow: 0 0 28px rgba(0,212,255,.12);
  transform: translateY(-2px);
}
.kpi-label {
  font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(0,212,255,.7); margin-bottom:6px;
}
.kpi-value { font-size:30px; font-weight:800; color:#fff; line-height:1.1; }
.kpi-sub   { font-size:11px; color:rgba(255,255,255,.4); margin-top:3px; }

/* Section headers */
.sec-hdr {
  font-size:18px; font-weight:700; color:#fff;
  border-left:3px solid #00d4ff; padding-left:10px; margin:22px 0 14px;
}

/* Risk badges */
.badge-critical { background:#ff2d55; color:#fff; border-radius:5px; padding:2px 7px; font-size:11px; font-weight:700; }
.badge-high     { background:#ff9f0a; color:#000; border-radius:5px; padding:2px 7px; font-size:11px; font-weight:700; }
.badge-medium   { background:#ffd60a; color:#000; border-radius:5px; padding:2px 7px; font-size:11px; font-weight:700; }
.badge-low      { background:#30d158; color:#000; border-radius:5px; padding:2px 7px; font-size:11px; font-weight:700; }

::-webkit-scrollbar       { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,.02); }
::-webkit-scrollbar-thumb { background:rgba(0,212,255,.3); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Module imports ───────────────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

try:
    from ml_model import (
        load_soh_model, load_rul_model, load_failure_model, load_thermal_model,
        predict_soh, predict_rul, predict_failure_proba,
        predict_thermal_risk, predict_thermal_proba,
        SOH_FEATURES, RUL_FEATURES, THERMAL_FEATURES, FAILURE_FEATURES,
        RISK_ORDER, MODELS_DIR
    )
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#ffffff",
    margin=dict(l=0,r=0,t=30,b=0),
)
GRID = dict(gridcolor="rgba(255,255,255,0.06)")

# ── Caching ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_kpis():
    with DatabaseManager() as db: return db.get_fleet_kpis()

@st.cache_data(ttl=300, show_spinner=False)
def load_degradation(n=30):
    with DatabaseManager() as db: return db.get_degradation_curves(sample_n=n)

@st.cache_data(ttl=300, show_spinner=False)
def load_station_demand():
    with DatabaseManager() as db: return db.get_station_demand()

@st.cache_data(ttl=300, show_spinner=False)
def load_thermal_alerts():
    with DatabaseManager() as db: return db.get_thermal_alerts(("HIGH","CRITICAL"))

@st.cache_data(ttl=600, show_spinner=False)
def load_vehicle_list():
    with DatabaseManager() as db: return db.get_vehicle_list()

@st.cache_data(ttl=300, show_spinner=False)
def load_vehicle_history(vid):
    with DatabaseManager() as db: return db.get_vehicle_history(vid)

@st.cache_resource(show_spinner=False)
def load_models():
    return (load_soh_model(), load_rul_model(),
            load_failure_model(), load_thermal_model())

@st.cache_data(ttl=600, show_spinner=False)
def load_metadata():
    p = os.path.join(MODELS_DIR if ML_AVAILABLE else "models","feature_metadata.json")
    return json.load(open(p)) if os.path.exists(p) else {}

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px;">
      <div style="font-size:36px;">⚡</div>
      <div style="font-size:15px;font-weight:800;color:#00d4ff;letter-spacing:1px;">
        EV FLEET INTELLIGENCE
      </div>
      <div style="font-size:10px;color:rgba(255,255,255,.38);margin-top:4px;">
        EVIoT Predictive Maintenance · 175K Records
      </div>
    </div>
    <hr style="border-color:rgba(0,212,255,.18);">
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Fleet Overview",
        "📉  Degradation & Health",
        "🗺️  Station Heatmap",
        "🌡️  Thermal Safety Log",
        "🤖  Predictive Analytics",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(0,212,255,.18);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:rgba(255,255,255,.28);text-align:center;padding:6px;">
      Real EVIoT Dataset · 2020-2025<br>SQLite · 4 ML Models
    </div>""", unsafe_allow_html=True)

# ── DB guard ─────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "fleet_telemetry.db")
if not os.path.exists(DB_PATH):
    st.error("""
    **Database not found.** Run:
    ```bash
    python data_loader.py   # Download & ingest real EVIoT dataset
    python ml_model.py      # Train 4 ML models
    ```
    """)
    st.stop()

def plot(fig, height=350):
    fig.update_layout(height=height, **PLOT_THEME)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, use_container_width=True)

def kpi(col, label, value, sub, color="#ffffff"):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{color};">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Fleet Overview
# ════════════════════════════════════════════════════════════════════════════════
if "Fleet Overview" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">
      ⚡ Fleet Command Center
    </h1>
    <p style="color:rgba(255,255,255,.4);font-size:13px;margin-bottom:20px;">
      175,393 real-world IoT records · 2020–2025 operational window
    </p>""", unsafe_allow_html=True)

    kpis = load_kpis()
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1,"FLEET SIZE",       f"{kpis['fleet_size']:,}",                 "vehicles")
    kpi(c2,"TOTAL DISTANCE",   f"{kpis['total_distance_km']/1e6:.2f}M",   "km driven","#00d4ff")
    kpi(c3,"AVG CHARGING EFF", f"{kpis['avg_charging_eff']:.1f}%",        "efficiency")
    kpi(c4,"CARBON OFFSET",    f"{kpis['carbon_offset_t']:.0f}t",         "CO2 saved","#30d158")
    kpi(c5,"AVG BATTERY SOH",  f"{kpis['avg_soh']:.1f}%",                 "state of health")
    kpi(c6,"FAILURE EVENTS",   f"{kpis['failure_events']:,}",             "logged","#ff2d55")
    st.markdown("<br>", unsafe_allow_html=True)

    # SoH + Component Health distribution
    left, right = st.columns([3,2])
    with left:
        sec("Fleet SoH Distribution")
        deg = load_degradation(n=200)
        if not deg.empty:
            latest = deg.sort_values("Timestamp").groupby("vehicle_id").last().reset_index()
            fig = px.histogram(latest, x="SoH", nbins=30,
                               color_discrete_sequence=["#00d4ff"],
                               labels={"SoH":"State of Health (%)"},
                               template="plotly_dark")
            fig.add_vline(x=kpis["avg_soh"], line_dash="dash", line_color="#ffd60a",
                          annotation_text="Fleet Avg", annotation_font_color="#ffd60a")
            fig.add_vline(x=80, line_dash="dot", line_color="#ff2d55",
                          annotation_text="EOL (80%)", annotation_font_color="#ff2d55")
            plot(fig, 290)

    with right:
        sec("Thermal Risk Split")
        with DatabaseManager() as db:
            rc = db.query_df("SELECT thermal_risk, COUNT(*) as cnt FROM fleet_telemetry GROUP BY thermal_risk")
        fig2 = px.pie(rc, names="thermal_risk", values="cnt",
                      color="thermal_risk", hole=0.58,
                      color_discrete_map={"LOW":"#30d158","MEDIUM":"#ffd60a",
                                          "HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                      template="plotly_dark")
        fig2.update_traces(textinfo="percent+label", textfont_size=11)
        plot(fig2, 290)

    # Failure probability trend + Component Health by vehicle
    sec("Failure Events Over Time")
    with DatabaseManager() as db:
        fp_df = db.query_df("""
        SELECT substr(Timestamp,1,7) as month,
               SUM(Failure_Probability) as failures,
               COUNT(*) as total
        FROM fleet_telemetry GROUP BY month ORDER BY month
        """)
    if not fp_df.empty:
        fp_df["failure_rate"] = (fp_df["failures"] / fp_df["total"] * 100).round(3)
        fig3 = px.area(fp_df, x="month", y="failure_rate",
                       labels={"month":"Month","failure_rate":"Failure Rate (%)"},
                       color_discrete_sequence=["#ff2d55"], template="plotly_dark")
        fig3.update_traces(fill="tozeroy", fillcolor="rgba(255,45,85,0.15)")
        plot(fig3, 240)

    # RUL + Component Health KPIs
    sec("Fleet Health Deep Metrics")
    m1,m2,m3,m4 = st.columns(4)
    kpi(m1,"AVG RUL",            f"{kpis['avg_rul']:.0f}",      "days remaining","#7c3aed")
    kpi(m2,"AVG COMPONENT HEALTH",f"{kpis['avg_health']:.3f}",  "0→1 score","#00d4ff")
    kpi(m3,"MIN BATTERY SOH",    f"{kpis['min_soh']:.1f}%",     "worst vehicle","#ff9f0a")
    kpi(m4,"DATA RECORDS",       "175,393",                      "real IoT samples","#30d158")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Degradation & Health
# ════════════════════════════════════════════════════════════════════════════════
elif "Degradation" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">
      📉 Degradation & Component Health
    </h1>
    <p style="color:rgba(255,255,255,.4);font-size:13px;margin-bottom:20px;">
      Real charge-cycle degradation curves and multi-system health scoring
    </p>""", unsafe_allow_html=True)

    n = st.slider("Vehicles to display", 5, 60, 25, 5)
    deg = load_degradation(n=n)
    if not deg.empty:
        deg["Timestamp"] = pd.to_datetime(deg["Timestamp"])

        sec("SoH vs Charge Cycles")
        fig = px.line(deg.sort_values("Charge_Cycles"),
                      x="Charge_Cycles", y="SoH", color="vehicle_id",
                      labels={"Charge_Cycles":"Charge Cycles","SoH":"SoH (%)","vehicle_id":"Vehicle"},
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig.add_hline(y=80, line_dash="dash", line_color="#ff2d55",
                      annotation_text="EOL 80%", annotation_font_color="#ff2d55")
        fig.add_hline(y=90, line_dash="dot", line_color="#ffd60a",
                      annotation_text="Warning 90%", annotation_font_color="#ffd60a")
        fig.update_layout(showlegend=False)
        plot(fig, 400)

        sec("Component Health Score Distribution")
        fig2 = px.histogram(deg, x="Component_Health_Score", nbins=30,
                            color_discrete_sequence=["#7c3aed"],
                            labels={"Component_Health_Score":"Component Health (0-1)"},
                            template="plotly_dark")
        plot(fig2, 260)

    sec("Single Vehicle Deep-Dive")
    vehicles = load_vehicle_list()
    sel = st.selectbox("Select Vehicle", vehicles, key="deg_veh")
    vh  = load_vehicle_history(sel)
    if not vh.empty:
        vh["Timestamp"] = pd.to_datetime(vh["Timestamp"])
        vh = vh.sort_values("Timestamp")

        col1,col2 = st.columns(2)
        with col1:
            fig3 = px.line(vh, x="Timestamp", y="SoH",
                           color_discrete_sequence=["#00d4ff"],
                           title=f"{sel} — SoH Over Time",
                           template="plotly_dark")
            plot(fig3, 270)
        with col2:
            fig4 = px.line(vh, x="Timestamp", y="RUL",
                           color_discrete_sequence=["#7c3aed"],
                           title=f"{sel} — Remaining Useful Life (days)",
                           template="plotly_dark")
            plot(fig4, 270)

        col3,col4 = st.columns(2)
        with col3:
            fig5 = px.scatter(vh, x="Battery_Temperature", y="SoH",
                              color="thermal_risk",
                              color_discrete_map={"LOW":"#30d158","MEDIUM":"#ffd60a",
                                                  "HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                              title=f"{sel} — Battery Temp vs SoH",
                              template="plotly_dark")
            plot(fig5, 270)
        with col4:
            fig6 = px.line(vh, x="Timestamp", y="Component_Health_Score",
                           color_discrete_sequence=["#30d158"],
                           title=f"{sel} — Component Health Score",
                           template="plotly_dark")
            plot(fig6, 270)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Station Heatmap
# ════════════════════════════════════════════════════════════════════════════════
elif "Heatmap" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">
      🗺️ Charging Station Demand Heatmap
    </h1>
    <p style="color:rgba(255,255,255,.4);font-size:13px;margin-bottom:20px;">
      Fleet charging node utilisation · US-West grid
    </p>""", unsafe_allow_html=True)

    sd = load_station_demand()
    if sd.empty:
        st.warning("No station data.")
    else:
        fig = px.scatter_mapbox(
            sd, lat="station_lat", lon="station_lon",
            size="charge_sessions", color="total_energy_kwh",
            hover_name="station_name",
            hover_data={"charge_sessions":True,"total_energy_kwh":":.0f",
                        "station_lat":False,"station_lon":False},
            color_continuous_scale="Turbo", size_max=55,
            zoom=3.5, center={"lat":39.0,"lon":-115.0},
            mapbox_style="carto-darkmatter",
            labels={"total_energy_kwh":"Energy (kWh)"},
            template="plotly_dark",
        )
        fig.update_layout(height=520, **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

        sec("Station Leaderboard")
        c1,c2 = st.columns(2)
        with c1:
            f1 = px.bar(sd.sort_values("charge_sessions",ascending=True),
                        x="charge_sessions",y="station_name",orientation="h",
                        color="charge_sessions",color_continuous_scale="Blues",
                        title="Charging Sessions",template="plotly_dark")
            f1.update_layout(showlegend=False,coloraxis_showscale=False)
            plot(f1, 300)
        with c2:
            f2 = px.bar(sd.sort_values("total_energy_kwh",ascending=True),
                        x="total_energy_kwh",y="station_name",orientation="h",
                        color="total_energy_kwh",color_continuous_scale="Reds",
                        title="Energy Dispensed (kWh)",template="plotly_dark")
            f2.update_layout(showlegend=False,coloraxis_showscale=False)
            plot(f2, 300)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Thermal Safety Log
# ════════════════════════════════════════════════════════════════════════════════
elif "Thermal" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">
      🌡️ Thermal Safety Warning Log
    </h1>
    <p style="color:rgba(255,255,255,.4);font-size:13px;margin-bottom:20px;">
      Battery & motor overheating incidents from real IoT sensor data
    </p>""", unsafe_allow_html=True)

    alerts = load_thermal_alerts()
    if alerts.empty:
        st.success("No HIGH/CRITICAL thermal events found.")
    else:
        alerts["Timestamp"] = pd.to_datetime(alerts["Timestamp"])

        c1,c2,c3,c4 = st.columns(4)
        nc = len(alerts[alerts["thermal_risk"]=="CRITICAL"])
        nh = len(alerts[alerts["thermal_risk"]=="HIGH"])
        kpi(c1,"CRITICAL EVENTS", nc,                                    "events","#ff2d55")
        kpi(c2,"HIGH EVENTS",     nh,                                    "events","#ff9f0a")
        kpi(c3,"VEHICLES AFFECTED",alerts["vehicle_id"].nunique(),       "vehicles","#ffd60a")
        kpi(c4,"MAX CELL TEMP",   f"{alerts['Battery_Temperature'].max():.1f}°C","recorded","#00d4ff")
        st.markdown("<br>", unsafe_allow_html=True)

        sec("Thermal Events Timeline")
        alerts["date"] = alerts["Timestamp"].dt.date
        daily = alerts.groupby(["date","thermal_risk"]).size().reset_index(name="count")
        fig = px.area(daily, x="date", y="count", color="thermal_risk",
                      color_discrete_map={"HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                      template="plotly_dark")
        plot(fig, 260)

        sec("Sensor Distributions at Alert")
        cl,cr = st.columns(2)
        with cl:
            f2 = px.violin(alerts, x="thermal_risk", y="Battery_Temperature",
                           color="thermal_risk", box=True,
                           color_discrete_map={"HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                           labels={"Battery_Temperature":"Battery Temp (°C)"},
                           template="plotly_dark")
            f2.update_layout(showlegend=False)
            plot(f2, 300)
        with cr:
            f3 = px.violin(alerts, x="thermal_risk", y="Motor_Temperature",
                           color="thermal_risk", box=True,
                           color_discrete_map={"HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                           labels={"Motor_Temperature":"Motor Temp (°C)"},
                           template="plotly_dark")
            f3.update_layout(showlegend=False)
            plot(f3, 300)

        sec("Top Affected Vehicles")
        top = alerts.groupby("vehicle_id").size().reset_index(name="events").sort_values("events",ascending=False).head(12)
        f4 = px.bar(top, x="events",y="vehicle_id",orientation="h",
                    color="events",color_continuous_scale="Reds",template="plotly_dark")
        f4.update_layout(showlegend=False,coloraxis_showscale=False)
        plot(f4, 320)

        sec("Recent Alert Log")
        risk_f = st.multiselect("Filter", ["HIGH","CRITICAL"], default=["HIGH","CRITICAL"])
        tbl = alerts[alerts["thermal_risk"].isin(risk_f)].sort_values("Timestamp",ascending=False).head(300).copy()
        def badge(r):
            return f'<span class="badge-{r.lower()}">{r}</span>'
        tbl["Risk"] = tbl["thermal_risk"].apply(badge)
        disp = tbl[["vehicle_id","Timestamp","Battery_Temperature","Motor_Temperature",
                    "Battery_Current","SoC","Component_Health_Score","Risk"]]
        disp.columns = ["Vehicle","Timestamp","Batt Temp °C","Motor Temp °C","Current A","SoC %","Health","Risk"]
        st.write(disp.to_html(escape=False,index=False), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Predictive Analytics
# ════════════════════════════════════════════════════════════════════════════════
elif "Predictive" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">
      🤖 Predictive Battery Analytics
    </h1>
    <p style="color:rgba(255,255,255,.4);font-size:13px;margin-bottom:20px;">
      4-model ML panel trained on 175K real EVIoT records
    </p>""", unsafe_allow_html=True)

    models_ok = all(
        os.path.exists(os.path.join("models", f))
        for f in ["soh_regressor.pkl","rul_regressor.pkl",
                  "failure_classifier.pkl","thermal_classifier.pkl"]
    )
    if not models_ok or not ML_AVAILABLE:
        st.warning("Train models first: `python ml_model.py`")
        st.stop()

    soh_m, rul_m, fail_m, therm_m = load_models()
    meta = load_metadata()

    sec("Model Performance")
    mc1,mc2,mc3,mc4 = st.columns(4)
    sm = meta.get("soh_regressor",{})
    rm = meta.get("rul_regressor",{})
    fm = meta.get("failure_classifier",{})
    tm = meta.get("thermal_classifier",{})
    kpi(mc1,"SOH MODEL R2",    f"{sm.get('r2',0):.4f}",            "random forest","#00d4ff")
    kpi(mc2,"RUL MODEL R2",    f"{rm.get('r2',0):.4f}",            "gradient boost","#7c3aed")
    kpi(mc3,"FAILURE ACCURACY",f"{fm.get('accuracy',0)*100:.1f}%", "classifier","#ff9f0a")
    kpi(mc4,"THERMAL ACCURACY",f"{tm.get('accuracy',0)*100:.1f}%", "classifier","#30d158")
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    if sm.get("importances"):
        sec("SoH Model — Feature Importance")
        imp_df = pd.DataFrame(list(sm["importances"].items()),
                              columns=["Feature","Importance"]).sort_values("Importance")
        fi = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues", template="plotly_dark")
        fi.update_layout(showlegend=False, coloraxis_showscale=False)
        plot(fi, 260)

    sec("Live Vehicle Risk Assessment")
    col_v, col_b = st.columns([3,1])
    with col_v:
        sel_v = st.selectbox("Load vehicle snapshot", load_vehicle_list(), key="pred_v")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        do_load = st.button("Load Latest Snapshot", use_container_width=True)

    # Defaults
    D = {
        "SoC":80.0, "SoH":88.0, "Battery_Voltage":360.0, "Battery_Current":-30.0,
        "Battery_Temperature":32.0, "Charge_Cycles":200.0, "Motor_Temperature":55.0,
        "Motor_Vibration":0.5, "Motor_Torque":110.0, "Motor_RPM":1800.0,
        "Power_Consumption":25.0, "Brake_Pad_Wear":0.2, "Ambient_Temperature":22.0,
        "Driving_Speed":60.0, "Distance_Traveled":50.0, "Route_Roughness":0.3,
        "Load_Weight":700.0, "Component_Health_Score":0.85, "RUL":200.0, "TTF":150.0,
    }
    if do_load:
        vh = load_vehicle_history(sel_v)
        if not vh.empty:
            row = vh.sort_values("Timestamp").iloc[-1]
            for k in D:
                if k in row.index and pd.notna(row[k]):
                    D[k] = float(row[k])
            st.success(f"Loaded snapshot for {sel_v}")

    with st.form("pred_form"):
        st.markdown("**Adjust sensor readings:**")
        r1 = st.columns(4)
        soc  = r1[0].number_input("SoC (%)",           0.0, 100.0, D["SoC"],        0.5)
        soh  = r1[1].number_input("SoH (%)",           0.0, 100.0, D["SoH"],        0.5)
        bv   = r1[2].number_input("Battery Voltage (V)",200.0,450.0,D["Battery_Voltage"],1.0)
        bc   = r1[3].number_input("Battery Current (A)",-250.0,250.0,D["Battery_Current"],1.0)

        r2 = st.columns(4)
        bt  = r2[0].number_input("Battery Temp (°C)", -10.0, 90.0, D["Battery_Temperature"], 0.5)
        mt  = r2[1].number_input("Motor Temp (°C)",    0.0, 150.0, D["Motor_Temperature"],   0.5)
        cc  = r2[2].number_input("Charge Cycles",       0.0,2000.0, D["Charge_Cycles"],     10.0)
        pc  = r2[3].number_input("Power (kW)",          0.0, 200.0, D["Power_Consumption"],  1.0)

        r3 = st.columns(4)
        mv  = r3[0].number_input("Motor Vibration",    0.0,  10.0, D["Motor_Vibration"],    0.1)
        bpw = r3[1].number_input("Brake Pad Wear",     0.0,   1.0, D["Brake_Pad_Wear"],     0.01)
        chs = r3[2].number_input("Component Health",   0.0,   1.0, D["Component_Health_Score"],0.01)
        rul_in = r3[3].number_input("Current RUL (days)", 0.0,500.0, D["RUL"],           5.0)

        r4 = st.columns(4)
        at  = r4[0].number_input("Ambient Temp (°C)", -20.0,55.0, D["Ambient_Temperature"], 0.5)
        ds  = r4[1].number_input("Driving Speed (km/h)",0.0,200.0, D["Driving_Speed"],    1.0)
        lw  = r4[2].number_input("Load Weight (kg)",   0.0,2000.0, D["Load_Weight"],      50.0)
        rr  = r4[3].number_input("Route Roughness",    0.0,   1.0, D["Route_Roughness"],  0.05)

        submitted = st.form_submit_button("Run All 4 Predictions", use_container_width=True)

    if submitted:
        inputs = {
            "SoC":soc,"SoH":soh,"Battery_Voltage":bv,"Battery_Current":bc,
            "Battery_Temperature":bt,"Charge_Cycles":cc,"Motor_Temperature":mt,
            "Motor_Vibration":mv,"Motor_Torque":D["Motor_Torque"],"Motor_RPM":D["Motor_RPM"],
            "Power_Consumption":pc,"Brake_Pad_Wear":bpw,"Ambient_Temperature":at,
            "Driving_Speed":ds,"Distance_Traveled":D["Distance_Traveled"],
            "Route_Roughness":rr,"Load_Weight":lw,
            "Component_Health_Score":chs,"RUL":rul_in,"TTF":D["TTF"],
        }

        pred_soh_v  = predict_soh(soh_m, inputs)
        rul_feats   = meta.get("rul_regressor",{}).get("features", RUL_FEATURES)
        pred_rul_v  = predict_rul(rul_m, inputs, rul_feats)
        fail_feats  = meta.get("failure_classifier",{}).get("features", FAILURE_FEATURES)
        pred_fail_v = predict_failure_proba(fail_m, inputs, fail_feats)
        pred_risk   = predict_thermal_risk(therm_m, inputs)
        pred_proba  = predict_thermal_proba(therm_m, inputs)

        st.markdown("---")
        st.markdown("### Prediction Results")

        soh_color  = "#30d158" if pred_soh_v>=90 else "#ffd60a" if pred_soh_v>=80 else "#ff9f0a" if pred_soh_v>=70 else "#ff2d55"
        risk_color = {"LOW":"#30d158","MEDIUM":"#ffd60a","HIGH":"#ff9f0a","CRITICAL":"#ff2d55"}.get(pred_risk,"#fff")
        fail_color = "#ff2d55" if pred_fail_v>0.5 else "#ff9f0a" if pred_fail_v>0.25 else "#30d158"

        rp1,rp2,rp3,rp4 = st.columns(4)
        for col,label,val,sub,col_c in [
            (rp1,"PREDICTED SoH",   f"{pred_soh_v:.1f}%",  "battery health",         soh_color),
            (rp2,"PREDICTED RUL",   f"{pred_rul_v:.0f}d",  "remaining useful life",  "#7c3aed"),
            (rp3,"FAILURE PROB",    f"{pred_fail_v*100:.1f}%","risk of failure",      fail_color),
            (rp4,"THERMAL RISK",    pred_risk,               "overheating class",     risk_color),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{col_c}55;">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value" style="color:{col_c};font-size:34px;">{val}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)

        # SoH gauge
        with gc1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=pred_soh_v,
                number={"suffix":"%","font":{"color":soh_color,"size":28}},
                title={"text":"SoH Gauge","font":{"color":"#fff","size":13}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#fff"},
                    "bar":{"color":soh_color},
                    "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(255,255,255,.08)",
                    "steps":[
                        {"range":[0, 70],"color":"rgba(255,45,85,.12)"},
                        {"range":[70,80],"color":"rgba(255,159,10,.12)"},
                        {"range":[80,90],"color":"rgba(255,214,10,.12)"},
                        {"range":[90,100],"color":"rgba(48,209,88,.12)"},
                    ],
                    "threshold":{"line":{"color":"#ff2d55","width":3},"thickness":0.75,"value":80},
                }
            ))
            gauge.update_layout(height=220,**PLOT_THEME)
            st.plotly_chart(gauge, use_container_width=True)

        # Thermal proba
        with gc2:
            prob_df = pd.DataFrame({
                "Risk":list(pred_proba.keys()),
                "Probability":[v*100 for v in pred_proba.values()]
            })
            fp2 = px.bar(prob_df, x="Risk", y="Probability",
                         color="Risk",
                         color_discrete_map={"LOW":"#30d158","MEDIUM":"#ffd60a",
                                             "HIGH":"#ff9f0a","CRITICAL":"#ff2d55"},
                         text="Probability", template="plotly_dark")
            fp2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fp2.update_layout(showlegend=False, yaxis_range=[0,115])
            plot(fp2, 220)
