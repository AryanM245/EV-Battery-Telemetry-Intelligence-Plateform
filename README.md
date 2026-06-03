# ⚡ EV Fleet Telemetry Analytics & Battery Health Predictive Dashboard

A production-ready Python data pipeline and executive analytics dashboard for monitoring and predicting battery health across a 500-vehicle EV fleet.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  telemetry_generator.py  →  db_manager.py  →  fleet_telemetry.db   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          AI/ML LAYER                                 │
│                ml_model.py  →  models/soh_model.pkl                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                               │
│                  app.py  (Streamlit Dashboard)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
EV tele/
├── app.py                   # Streamlit executive dashboard
├── telemetry_generator.py   # IoT time-series data simulator (500 EVs)
├── db_manager.py            # SQLite pipeline controller & data ingestion
├── ml_model.py              # Scikit-Learn SoH prediction & thermal classifier
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── README.md                # This file
├── fleet_telemetry.db       # Auto-generated SQLite database (after setup)
└── models/
    ├── soh_regressor.pkl    # Trained SoH regression model
    └── thermal_classifier.pkl  # Thermal risk classification model
```

---

## ⚙️ Tech Stack

| Layer         | Technology                          |
|---------------|-------------------------------------|
| Data Sim      | Python, NumPy, Faker                |
| Database      | SQLite3 (raw SQL)                   |
| ML Pipeline   | Scikit-Learn, Joblib                |
| Dashboard     | Streamlit, Plotly, Altair           |
| Container     | Docker                              |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Fleet Telemetry Data
```bash
python telemetry_generator.py
```
> Generates ~50,000 time-series records across 500 vehicles and ingests into SQLite.

### 3. Train ML Models
```bash
python ml_model.py
```
> Trains SoH regression + thermal risk classifier. Saves models to `models/`.

### 4. Launch Dashboard
```bash
streamlit run app.py
```
> Opens the executive dashboard at `http://localhost:8501`

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ev-telemetry-dashboard .

# Run container
docker run -p 8501:8501 ev-telemetry-dashboard
```

---

## 📊 Dashboard Features

- **KPI Cards** — Total fleet distance, charging efficiency, carbon offset
- **Degradation Curves** — Per-vehicle battery SoH over operational lifetime
- **Station Demand Heatmap** — Geographic charging station utilization
- **Thermal Safety Logs** — Real-time overheating risk alerts
- **Predictive Analytics** — ML-powered per-vehicle battery health prediction

---

## 🔬 ML Models

### State-of-Health (SoH) Regressor
- **Algorithm**: Random Forest Regressor
- **Features**: Voltage, temperature, current, SoC, cycle count, ambient temp
- **Target**: SoH (0–100%) degradation prediction

### Thermal Risk Classifier
- **Algorithm**: Gradient Boosting Classifier
- **Features**: Cell temperature, ambient temp, current draw, SoC
- **Target**: Overheating risk (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)

---

## 📄 License

MIT License — Free for commercial and personal use.
