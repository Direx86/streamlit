"""
Smart Home Energy Prediction - Streamlit App
Prediksi Konsumsi Energi Rumah Tangga menggunakan Random Forest, Gradient Boosting, dan LSTM.
Dataset: UCI Individual Household Electric Power Consumption.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import zipfile
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# ─── CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home Energy Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "backend" / "data"

DATASET_ZIP_PATH = MODELS_DIR / "individual+household+electric+power+consumption.zip"
DATASET_TXT_PATH = MODELS_DIR / "household_power_consumption.txt"

RF_MODEL_PATH = MODELS_DIR / "random_forest_regressor_TUNED.joblib"
GB_MODEL_PATH = MODELS_DIR / "gradient_boosting_regressor_TUNED.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
METRICS_CSV_PATH = MODELS_DIR / "test_metrics_default.csv"

SAMPLE_DATA_PATH = DATA_DIR / "sample_data.json"
FULL_TEST_DATA_PATH = DATA_DIR / "full_test_data.json"

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"

SCALER_MEAN = 1.086397
SCALER_STD = 0.929282

TREE_FEATURE_COLS = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "hour", "dayofweek", "is_weekend", "month",
    "Global_active_power_lag1", "Global_active_power_lag2",
    "Global_active_power_lag3", "Global_active_power_lag6",
    "Global_active_power_lag12", "Global_active_power_lag24",
    "Global_active_power_rollmean3", "Global_active_power_rollstd3",
    "Global_active_power_rollmean6", "Global_active_power_rollstd6",
    "Global_active_power_rollmean12", "Global_active_power_rollstd12",
    "Global_active_power_rollmean24", "Global_active_power_rollstd24",
]


# ─── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .metric-unit {
        font-size: 14px;
        color: #64748b;
        margin-left: 4px;
    }
    .info-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .model-badge-rf {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        display: inline-block;
    }
    .model-badge-gb {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid rgba(249, 115, 22, 0.3);
        display: inline-block;
    }
    .model-badge-lstm {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        display: inline-block;
    }
    .best-badge {
        background: rgba(234, 179, 8, 0.2);
        color: #facc15;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 6px;
    }
    .source-link {
        color: #38bdf8 !important;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .dataset-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 500;
        display: inline-block;
    }
    .conclusion-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
    }
    .conclusion-title {
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING (cached) ──────────────────────────────────
@st.cache_data
def extract_dataset():
    """Extract dataset from zip if not already extracted."""
    if not DATASET_TXT_PATH.exists() and DATASET_ZIP_PATH.exists():
        with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as z:
            z.extract("household_power_consumption.txt", MODELS_DIR)
    return DATASET_TXT_PATH.exists()


@st.cache_data
def load_sample_data():
    """Load pre-computed 168-hour sample data."""
    if SAMPLE_DATA_PATH.exists():
        with open(SAMPLE_DATA_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_full_test_data():
    """Load full test set (5121 points)."""
    if FULL_TEST_DATA_PATH.exists():
        with open(FULL_TEST_DATA_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_metrics_csv():
    """Load official metrics from CSV."""
    if METRICS_CSV_PATH.exists():
        return pd.read_csv(METRICS_CSV_PATH)
    return None


def compute_metrics(data_points):
    """Compute MAE, RMSE, R2 from data points."""
    models = [
        {"key": "rf_pred", "name": "RandomForest"},
        {"key": "gb_pred", "name": "GradientBoosting"},
        {"key": "lstm_pred", "name": "LSTM"},
    ]
    results = []
    for m in models:
        valid = [d for d in data_points if d.get(m["key"]) and d[m["key"]] != 0]
        if not valid:
            results.append({"model": m["name"], "MAE": 0, "RMSE": 0, "R2": 0})
            continue

        actual = np.array([d["actual"] for d in valid])
        pred = np.array([d[m["key"]] for d in valid])
        errors = actual - pred

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        results.append({"model": m["name"], "MAE": mae, "RMSE": rmse, "R2": r2})
    return results


# ─── CHART HELPER ────────────────────────────────────────────
def create_prediction_chart(data, title, show_models=None, height=500):
    """Create a Plotly line chart comparing model predictions."""
    if not data:
        return None

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig = go.Figure()

    if show_models is None:
        show_models = {"actual": True, "rf": True, "gb": True, "lstm": True}

    if show_models.get("actual", True) and "actual" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["actual"],
            name="Data Aktual", mode="lines",
            line=dict(color="#3b82f6", width=2.5),
        ))

    if show_models.get("rf", True) and "rf_pred" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["rf_pred"],
            name="Random Forest", mode="lines",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))

    if show_models.get("gb", True) and "gb_pred" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["gb_pred"],
            name="Gradient Boosting", mode="lines",
            line=dict(color="#f97316", width=2, dash="dot"),
        ))

    if show_models.get("lstm", True) and "lstm_pred" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["lstm_pred"],
            name="LSTM", mode="lines",
            line=dict(color="#ef4444", width=2, dash="dashdot"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
        xaxis=dict(
            title="Waktu",
            gridcolor="#334155",
            color="#94a3b8",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Global Active Power (kW)",
            gridcolor="#334155",
            color="#94a3b8",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(size=12),
        ),
        height=height,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )

    return fig


def render_metrics_table(metrics, title, subtitle=None):
    """Render a styled metrics comparison table."""
    if not metrics:
        return

    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

    # Find best values
    best_mae = min(metrics, key=lambda x: x["MAE"])["model"]
    best_rmse = min(metrics, key=lambda x: x["RMSE"])["model"]
    best_r2 = max(metrics, key=lambda x: x["R2"])["model"]

    badge_map = {
        "RandomForest": "model-badge-rf",
        "GradientBoosting": "model-badge-gb",
        "LSTM": "model-badge-lstm",
    }

    # Build table HTML
    rows_html = ""
    for m in metrics:
        badge_class = badge_map.get(m["model"], "")

        mae_str = f'{m["MAE"]:.4f}'
        rmse_str = f'{m["RMSE"]:.4f}'
        r2_str = f'{m["R2"]:.4f}'

        if m["model"] == best_mae:
            mae_str = f'<span style="color:#facc15;font-weight:700">{mae_str}</span><span class="best-badge">Best</span>'
        else:
            mae_str = f'<span style="color:#e2e8f0">{mae_str}</span>'

        if m["model"] == best_rmse:
            rmse_str = f'<span style="color:#facc15;font-weight:700">{rmse_str}</span><span class="best-badge">Best</span>'
        else:
            rmse_str = f'<span style="color:#e2e8f0">{rmse_str}</span>'

        if m["model"] == best_r2:
            r2_str = f'<span style="color:#facc15;font-weight:700">{r2_str}</span><span class="best-badge">Best</span>'
        else:
            r2_str = f'<span style="color:#e2e8f0">{r2_str}</span>'

        rows_html += f"""
        <tr style="border-top: 1px solid #334155;">
            <td style="padding:12px 16px;"><span class="{badge_class}">{m['model']}</span></td>
            <td style="padding:12px 16px; text-align:right; font-family:monospace; font-size:14px;">{mae_str}</td>
            <td style="padding:12px 16px; text-align:right; font-family:monospace; font-size:14px;">{rmse_str}</td>
            <td style="padding:12px 16px; text-align:right; font-family:monospace; font-size:14px;">{r2_str}</td>
        </tr>
        """

    table_html = f"""
    <div style="background:#1e293b; border:1px solid #334155; border-radius:16px; overflow:hidden;">
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">
                    <th style="text-align:left; padding:12px 16px;">Model</th>
                    <th style="text-align:right; padding:12px 16px;">MAE</th>
                    <th style="text-align:right; padding:12px 16px;">RMSE</th>
                    <th style="text-align:right; padding:12px 16px;">R&sup2;</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Smart Home Energy")
    st.markdown("**Prediksi Konsumsi Energi**")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["Dashboard", "Prediksi & Perbandingan", "Informasi Model"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f'<div class="dataset-badge">📊 Dataset: <a href="{UCI_DATASET_URL}" target="_blank" '
        f'style="color:#6ee7b7;">UCI Power Consumption</a></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.caption("Sumber: UCI Machine Learning Repository")
    st.caption(f"[Buka Dataset UCI]({UCI_DATASET_URL})")


# ─── EXTRACT DATASET ────────────────────────────────────────
extract_dataset()


# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Dashboard Smart Home")
    st.caption("Monitoring & Prediksi Konsumsi Energi Rumah Tangga")

    sample_data = load_sample_data()

    if sample_data and sample_data.get("data"):
        data = sample_data["data"]
        latest = data[-1]
        avg_val = sum(d["actual"] for d in data) / len(data)
        max_val = max(d["actual"] for d in data)
        min_val = min(d["actual"] for d in data)

        # Stat Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Konsumsi Terakhir", f"{latest['actual']:.2f} kW", help="Global Active Power")
        with col2:
            st.metric("Rata-rata (7 Hari)", f"{avg_val:.2f} kW", help=f"{len(data)} data point")
        with col3:
            st.metric("Konsumsi Tertinggi", f"{max_val:.2f} kW", help="Puncak pemakaian")
        with col4:
            st.metric("Konsumsi Terendah", f"{min_val:.2f} kW", help="Pemakaian minimum")

        st.markdown("---")

        # Dataset Variables
        st.subheader("Variabel Dataset UCI")
        vars_data = [
            ("Global Active Power", "Global_active_power", "kW", "Total daya aktif yang dikonsumsi rumah tangga (**TARGET prediksi**)"),
            ("Global Reactive Power", "Global_reactive_power", "kW", "Total daya reaktif yang dikonsumsi rumah tangga"),
            ("Voltage", "Voltage", "Volt", "Tegangan listrik rata-rata per menit"),
            ("Global Intensity", "Global_intensity", "Ampere", "Arus listrik rata-rata per menit"),
            ("Sub Metering 1", "Sub_metering_1", "Wh", "Dapur: dishwasher, oven, microwave"),
            ("Sub Metering 2", "Sub_metering_2", "Wh", "Laundry: mesin cuci, pengering, kulkas, lampu"),
            ("Sub Metering 3", "Sub_metering_3", "Wh", "Pemanas air listrik & AC"),
        ]

        cols = st.columns(4)
        for i, (name, var, unit, desc) in enumerate(vars_data):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#38bdf8; font-size:13px;">{name}</strong><br>
                    <code style="font-size:11px; color:#94a3b8;">{var} ({unit})</code><br>
                    <span style="font-size:12px; color:#cbd5e1;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-size:12px; color:#64748b;">Sumber: '
            f'<a href="{UCI_DATASET_URL}" target="_blank" class="source-link">'
            f'UCI Machine Learning Repository - Individual Household Electric Power Consumption</a></p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Data source badge
        st.markdown(
            f'<div class="dataset-badge">📊 Data Asli: '
            f'<a href="{UCI_DATASET_URL}" target="_blank" style="color:#6ee7b7;">'
            f'UCI Household Power Consumption</a> (Test Set: {len(data)} points)</div>',
            unsafe_allow_html=True,
        )

        # Chart
        fig = create_prediction_chart(
            data,
            f"Perbandingan Prediksi 3 Model ({len(data)} Jam)",
            height=450,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Metrics
        if sample_data.get("metrics"):
            render_metrics_table(
                sample_data["metrics"],
                "Perbandingan Metrik Evaluasi (Test Set)",
                "Semakin kecil MAE & RMSE = semakin baik. Semakin besar R² (mendekati 1) = semakin baik.",
            )
    else:
        st.warning("Data sampel tidak ditemukan. Jalankan `generate_sample_data.py` terlebih dahulu.")


# ═══════════════════════════════════════════════════════════
# PAGE: PREDIKSI & PERBANDINGAN
# ═══════════════════════════════════════════════════════════
elif page == "Prediksi & Perbandingan":
    st.title("Prediksi & Perbandingan")
    st.caption(
        "Bandingkan hasil prediksi Random Forest, Gradient Boosting, dan LSTM pada "
        f"[dataset UCI asli]({UCI_DATASET_URL})"
    )

    sample_data = load_sample_data()
    full_test_data = load_full_test_data()

    # Data source badge
    total_pts = len(full_test_data["data"]) if full_test_data else (len(sample_data["data"]) if sample_data else 0)
    st.markdown(
        f'<div class="dataset-badge">📊 Data Asli: '
        f'<a href="{UCI_DATASET_URL}" target="_blank" style="color:#6ee7b7;">'
        f'UCI Household Power Consumption</a> (Test Set: {total_pts} points)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Controls
    st.subheader("Kontrol Visualisasi")

    col_range, col_models = st.columns([1, 1])

    with col_range:
        st.markdown("**Rentang Waktu**")
        time_options = {"24 Jam": 24, "48 Jam": 48, "7 Hari": 168, "30 Hari": 720, "Full Test Set": 0}
        time_label = st.radio(
            "Pilih rentang waktu",
            list(time_options.keys()),
            index=2,
            horizontal=True,
            label_visibility="collapsed",
        )
        time_range = time_options[time_label]

    with col_models:
        st.markdown("**Tampilkan Model**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            show_actual = st.checkbox("Aktual", value=True)
        with mc2:
            show_rf = st.checkbox("RF", value=True)
        with mc3:
            show_gb = st.checkbox("GB", value=True)
        with mc4:
            show_lstm = st.checkbox("LSTM", value=True)

    show_models = {"actual": show_actual, "rf": show_rf, "gb": show_gb, "lstm": show_lstm}

    # Pick data source
    if time_range > 168 or time_range == 0:
        source = full_test_data
    else:
        source = sample_data

    if source and source.get("data"):
        data = source["data"]
        if time_range > 0:
            data = data[-time_range:]

        display_label = time_label

        # Main Chart
        fig = create_prediction_chart(
            data,
            f"Grafik Perbandingan - {display_label} ({len(data)} points)",
            show_models=show_models,
            height=500,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Error Analysis
        with st.expander(f"📊 Analisis Error ({len(data)} points)", expanded=False):
            models_analysis = [
                {"key": "rf_pred", "name": "Random Forest", "color": "#10b981"},
                {"key": "gb_pred", "name": "Gradient Boosting", "color": "#f97316"},
                {"key": "lstm_pred", "name": "LSTM", "color": "#ef4444"},
            ]

            ecols = st.columns(3)
            for i, m in enumerate(models_analysis):
                valid = [d for d in data if d.get(m["key"]) and d[m["key"]] != 0]
                if valid:
                    errors = [abs(d["actual"] - d[m["key"]]) for d in valid]
                    mae = sum(errors) / len(errors)
                    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
                    max_err = max(errors)

                    with ecols[i]:
                        st.markdown(f'<span style="color:{m["color"]}; font-weight:700; font-size:16px;">{m["name"]}</span>', unsafe_allow_html=True)
                        st.markdown(f"- **MAE:** `{mae:.4f}` kW")
                        st.markdown(f"- **RMSE:** `{rmse:.4f}` kW")
                        st.markdown(f"- **Max Error:** `{max_err:.4f}` kW")
                        st.markdown(f"- **Data Points:** `{len(valid)}`")

        # Dynamic metrics
        dynamic_metrics = compute_metrics(data)
        render_metrics_table(
            dynamic_metrics,
            f"Metrik Evaluasi - {display_label} ({len(data)} points)",
            "Dihitung ulang secara dinamis berdasarkan rentang waktu yang dipilih.",
        )

        st.markdown("")

        # Official metrics from full test set
        if sample_data and sample_data.get("metrics"):
            render_metrics_table(
                sample_data["metrics"],
                "Metrik Evaluasi Resmi - Full Test Set (5.121 points)",
                "Metrik tetap dari evaluasi seluruh test set (15% data). Ini adalah angka yang dilaporkan di skripsi.",
            )

        st.markdown("")

        # ─── PENJELASAN METRIK EVALUASI ──────────────────────
        st.markdown("---")
        st.subheader("Penjelasan Metrik Evaluasi")

        st.markdown("""
        <div class="conclusion-box">
            <h4 style="color:#38bdf8; margin-bottom:16px;">Apa itu MAE, RMSE, dan R²?</h4>

            <div style="margin-bottom:20px;">
                <h5 style="color:#34d399; margin-bottom:8px;">1. MAE (Mean Absolute Error) — Rata-rata Kesalahan Absolut</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.7;">
                    MAE mengukur <strong>rata-rata selisih absolut</strong> antara nilai prediksi dan nilai aktual.
                    Rumus: MAE = (1/n) × Σ|yᵢ - ŷᵢ|<br><br>
                    <strong>Interpretasi:</strong> MAE menunjukkan seberapa besar rata-rata kesalahan prediksi model
                    dalam satuan asli (kW). Semakin kecil MAE, semakin akurat model.
                    Misalnya, MAE = 0.0158 berarti rata-rata prediksi model meleset sebesar 0.0158 kW dari nilai sebenarnya.
                    <br><br>
                    <strong>Kelebihan MAE:</strong> Mudah diinterpretasikan karena menggunakan satuan yang sama dengan data asli,
                    dan tidak terlalu sensitif terhadap outlier (pencilan data).
                </p>
            </div>

            <div style="margin-bottom:20px;">
                <h5 style="color:#fb923c; margin-bottom:8px;">2. RMSE (Root Mean Squared Error) — Akar Rata-rata Kesalahan Kuadrat</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.7;">
                    RMSE mengukur <strong>akar dari rata-rata kuadrat selisih</strong> antara prediksi dan nilai aktual.
                    Rumus: RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]<br><br>
                    <strong>Interpretasi:</strong> RMSE juga dalam satuan yang sama (kW), namun memberikan
                    <strong>penalti lebih besar untuk kesalahan yang besar</strong> dibandingkan MAE.
                    Jika RMSE jauh lebih besar dari MAE, berarti ada beberapa prediksi yang meleset jauh (outlier error).
                    Semakin kecil RMSE, semakin konsisten model dalam memprediksi.
                    <br><br>
                    <strong>Perbandingan MAE vs RMSE:</strong> Jika MAE dan RMSE bernilai hampir sama (seperti pada RF dan GB),
                    berarti model memprediksi secara konsisten tanpa kesalahan ekstrem.
                </p>
            </div>

            <div style="margin-bottom:8px;">
                <h5 style="color:#f87171; margin-bottom:8px;">3. R² (R-Squared / Koefisien Determinasi)</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.7;">
                    R² mengukur <strong>proporsi variabilitas data yang dapat dijelaskan oleh model</strong>.
                    Rumus: R² = 1 - (SS_res / SS_tot), dimana SS_res = Σ(yᵢ - ŷᵢ)² dan SS_tot = Σ(yᵢ - ȳ)²<br><br>
                    <strong>Interpretasi:</strong> Nilai R² berkisar dari 0 hingga 1 (bisa negatif jika model sangat buruk).<br>
                    • <strong>R² = 1.0:</strong> Model sempurna — prediksi 100% cocok dengan data aktual<br>
                    • <strong>R² = 0.9989:</strong> Model menjelaskan 99.89% variasi data — sangat akurat<br>
                    • <strong>R² = 0.5243:</strong> Model menjelaskan 52.43% variasi data — cukup, namun banyak variasi yang tidak tertangkap<br>
                    • <strong>R² = 0.0:</strong> Model tidak lebih baik dari sekadar menghitung rata-rata<br><br>
                    <strong>Kelebihan R²:</strong> Memberikan gambaran menyeluruh tentang seberapa baik model mencocokkan data,
                    tidak bergantung pada skala/satuan data.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ─── INTERPRETASI HASIL DETAIL ───────────────────────
        st.subheader("Interpretasi Hasil")

        # Get official metrics for reference
        official = sample_data["metrics"] if sample_data and sample_data.get("metrics") else []
        rf_m = next((m for m in official if m["model"] == "RandomForest"), {})
        gb_m = next((m for m in official if m["model"] == "GradientBoosting"), {})
        lstm_m = next((m for m in official if m["model"] == "LSTM"), {})

        st.markdown(f"""
        <div class="conclusion-box">
            <div class="conclusion-title">📊 Analisis Performa Model</div>

            <div style="margin-bottom:20px; padding:16px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:12px;">
                <h5 style="color:#34d399; margin-bottom:10px;">1. Random Forest (RF)</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    • <strong>MAE = {rf_m.get('MAE', 0.0166):.4f} kW</strong> — Rata-rata kesalahan prediksi hanya ~0.017 kW,
                    artinya prediksi hampir selalu sangat dekat dengan nilai aktual.<br>
                    • <strong>RMSE = {rf_m.get('RMSE', 0.0249):.4f} kW</strong> — Kesalahan kuadrat juga sangat kecil dan
                    mendekati MAE, menunjukkan <strong>tidak ada kesalahan ekstrem</strong> (prediksi konsisten).<br>
                    • <strong>R² = {rf_m.get('R2', 0.9988):.4f}</strong> — Model mampu menjelaskan <strong>{rf_m.get('R2', 0.9988)*100:.2f}%</strong>
                    variasi data konsumsi energi. Ini berarti hampir seluruh pola konsumsi berhasil ditangkap oleh model.<br>
                    • <strong>Mengapa bagus?</strong> RF menggunakan 24 fitur termasuk lag features (nilai konsumsi 1-24 jam sebelumnya)
                    dan rolling statistics (rata-rata & standar deviasi bergerak), sehingga model memiliki konteks historis yang kaya.
                </p>
            </div>

            <div style="margin-bottom:20px; padding:16px; background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.2); border-radius:12px;">
                <h5 style="color:#fb923c; margin-bottom:10px;">2. Gradient Boosting (GB)</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    • <strong>MAE = {gb_m.get('MAE', 0.0158):.4f} kW</strong> — Rata-rata kesalahan prediksi paling kecil
                    di antara ketiga model, menunjukkan akurasi tertinggi.<br>
                    • <strong>RMSE = {gb_m.get('RMSE', 0.0239):.4f} kW</strong> — Juga paling kecil, berarti
                    GB paling konsisten dan memiliki <strong>kesalahan paling rendah</strong> secara keseluruhan.<br>
                    • <strong>R² = {gb_m.get('R2', 0.9989):.4f}</strong> — Nilai tertinggi,
                    menjelaskan <strong>{gb_m.get('R2', 0.9989)*100:.2f}%</strong> variasi data.
                    GB unggul karena metode boosting memperbaiki kesalahan secara iteratif.<br>
                    • <strong>Mengapa terbaik?</strong> GB membangun tree secara sekuensial — setiap tree baru
                    mempelajari kesalahan tree sebelumnya (residual learning). Ini membuat model
                    secara bertahap mengoreksi dan meminimalkan error. Ditambah dengan 24 fitur yang sama
                    seperti RF, hasilnya menjadi sangat optimal.
                </p>
            </div>

            <div style="margin-bottom:20px; padding:16px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px;">
                <h5 style="color:#f87171; margin-bottom:10px;">3. LSTM (Long Short-Term Memory)</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    • <strong>MAE = {lstm_m.get('MAE', 0.3462):.4f} kW</strong> — Rata-rata kesalahan ~0.35 kW,
                    sekitar <strong>22x lebih besar</strong> dari Gradient Boosting.<br>
                    • <strong>RMSE = {lstm_m.get('RMSE', 0.4896):.4f} kW</strong> — Jauh lebih besar dari MAE,
                    menunjukkan adanya beberapa prediksi yang <strong>meleset cukup jauh</strong> dari nilai aktual.<br>
                    • <strong>R² = {lstm_m.get('R2', 0.5243):.4f}</strong> — Hanya menjelaskan
                    <strong>{lstm_m.get('R2', 0.5243)*100:.2f}%</strong> variasi data.
                    Hampir setengah variasi konsumsi energi tidak tertangkap oleh model.<br>
                    • <strong>Mengapa lebih rendah?</strong> LSTM hanya menggunakan <strong>1 fitur</strong>
                    (Global_active_power saja, tanpa sensor lain atau fitur kalender), dengan window 24 jam.
                    Tanpa informasi kontekstual seperti hari kerja/weekend, jam, atau sub-metering,
                    LSTM kesulitan menangkap pola lengkap. Namun, LSTM tetap mampu menangkap
                    <strong>tren umum</strong> naik-turunnya konsumsi energi harian.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ─── PERBANDINGAN & KESIMPULAN ───────────────────────
        st.markdown(f"""
        <div class="conclusion-box">
            <div class="conclusion-title">🏆 Perbandingan RF vs GB: Mana yang Terbaik?</div>

            <div style="margin-bottom:16px;">
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    Dari ketiga metrik evaluasi pada full test set (5.121 data points), berikut perbandingannya:
                </p>
            </div>

            <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
                <tr style="border-bottom:1px solid #475569;">
                    <th style="text-align:left; padding:10px; color:#94a3b8; font-size:13px;">Metrik</th>
                    <th style="text-align:center; padding:10px; color:#34d399; font-size:13px;">Random Forest</th>
                    <th style="text-align:center; padding:10px; color:#fb923c; font-size:13px;">Gradient Boosting</th>
                    <th style="text-align:center; padding:10px; color:#94a3b8; font-size:13px;">Pemenang</th>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px; color:#e2e8f0;">MAE (↓ lebih baik)</td>
                    <td style="padding:10px; text-align:center; color:#e2e8f0; font-family:monospace;">{rf_m.get('MAE', 0.0166):.4f}</td>
                    <td style="padding:10px; text-align:center; color:#facc15; font-family:monospace; font-weight:700;">{gb_m.get('MAE', 0.0158):.4f}</td>
                    <td style="padding:10px; text-align:center;"><span class="model-badge-gb">GB</span></td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px; color:#e2e8f0;">RMSE (↓ lebih baik)</td>
                    <td style="padding:10px; text-align:center; color:#e2e8f0; font-family:monospace;">{rf_m.get('RMSE', 0.0249):.4f}</td>
                    <td style="padding:10px; text-align:center; color:#facc15; font-family:monospace; font-weight:700;">{gb_m.get('RMSE', 0.0239):.4f}</td>
                    <td style="padding:10px; text-align:center;"><span class="model-badge-gb">GB</span></td>
                </tr>
                <tr>
                    <td style="padding:10px; color:#e2e8f0;">R² (↑ lebih baik)</td>
                    <td style="padding:10px; text-align:center; color:#e2e8f0; font-family:monospace;">{rf_m.get('R2', 0.9988):.4f}</td>
                    <td style="padding:10px; text-align:center; color:#facc15; font-family:monospace; font-weight:700;">{gb_m.get('R2', 0.9989):.4f}</td>
                    <td style="padding:10px; text-align:center;"><span class="model-badge-gb">GB</span></td>
                </tr>
            </table>

            <div style="padding:16px; background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.3); border-radius:12px; margin-bottom:16px;">
                <h5 style="color:#fb923c; margin-bottom:10px;">🥇 Kesimpulan: Gradient Boosting adalah Model Terbaik</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    <strong>Gradient Boosting mengungguli semua model di ketiga metrik evaluasi.</strong><br><br>
                    <strong>1. MAE terendah ({gb_m.get('MAE', 0.0158):.4f} vs {rf_m.get('MAE', 0.0166):.4f}):</strong>
                    GB memiliki rata-rata error yang lebih kecil ~5% dibanding RF.
                    Artinya secara rata-rata, prediksi GB lebih mendekati nilai aktual.<br><br>
                    <strong>2. RMSE terendah ({gb_m.get('RMSE', 0.0239):.4f} vs {rf_m.get('RMSE', 0.0249):.4f}):</strong>
                    GB juga lebih konsisten — kesalahan besar (outlier error) lebih jarang terjadi
                    dibanding RF. Selisih RMSE ~4% mengonfirmasi keunggulan GB.<br><br>
                    <strong>3. R² tertinggi ({gb_m.get('R2', 0.9989):.4f} vs {rf_m.get('R2', 0.9988):.4f}):</strong>
                    GB mampu menjelaskan {gb_m.get('R2', 0.9989)*100:.2f}% variasi data, sedikit lebih tinggi
                    dari RF ({rf_m.get('R2', 0.9988)*100:.2f}%). Keduanya sangat mendekati 1.0,
                    namun GB tetap lebih unggul.<br><br>
                    <strong>Alasan GB lebih unggul:</strong> Gradient Boosting menggunakan strategi <em>sequential learning</em>
                    (boosting), di mana setiap tree baru secara eksplisit mempelajari dan mengoreksi kesalahan
                    tree sebelumnya. Berbeda dengan Random Forest yang melatih tree secara independen (bagging),
                    GB secara iteratif memperbaiki prediksi sehingga mencapai akurasi yang sedikit lebih tinggi.
                    Selain itu, ukuran model GB (~330 KB) jauh lebih kecil dibanding RF (~50 MB),
                    menjadikannya lebih efisien untuk deployment.
                </p>
            </div>

            <div style="padding:16px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px;">
                <h5 style="color:#f87171; margin-bottom:10px;">Catatan tentang LSTM</h5>
                <p style="color:#cbd5e1; font-size:14px; line-height:1.8;">
                    LSTM memiliki performa yang jauh lebih rendah (R² = {lstm_m.get('R2', 0.5243):.4f}) <strong>bukan karena
                    arsitekturnya buruk</strong>, melainkan karena keterbatasan input:
                    LSTM hanya menggunakan 1 fitur (Global_active_power) tanpa fitur tambahan
                    seperti sensor lain, informasi waktu, atau lag/rolling features yang digunakan oleh RF dan GB (24 fitur).
                    <br><br>
                    Jika LSTM diberikan fitur yang sama (multivariate LSTM), performanya berpotensi
                    meningkat signifikan. Namun dalam eksperimen ini, LSTM berfungsi sebagai
                    <strong>baseline univariate</strong> untuk menunjukkan pentingnya feature engineering
                    dalam prediksi time series.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            f'<p style="font-size:12px; color:#64748b;">Semua data berasal dari '
            f'<strong><a href="{UCI_DATASET_URL}" target="_blank" style="color:#94a3b8;">'
            f'dataset UCI asli</a></strong> (household_power_consumption.txt). '
            f'Periode test set: April - November 2010.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Data tidak ditemukan. Pastikan file JSON sudah di-generate.")


# ═══════════════════════════════════════════════════════════
# PAGE: INFORMASI MODEL
# ═══════════════════════════════════════════════════════════
elif page == "Informasi Model":
    st.title("Informasi Model")
    st.caption("Detail arsitektur, parameter, dan karakteristik masing-masing model prediksi")

    # Model cards
    models_info = [
        {
            "name": "Random Forest",
            "type": "Ensemble (Bagging)",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.08)",
            "border": "rgba(16, 185, 129, 0.25)",
            "description": (
                "Ensemble dari banyak Decision Tree yang dilatih secara paralel dengan teknik bagging. "
                "Setiap tree dilatih pada subset data acak, lalu hasil prediksi dirata-ratakan. "
                "Robust terhadap overfitting dan mampu menangkap hubungan non-linear."
            ),
            "parameters": {
                "n_estimators": "80-200 (tuned via RandomizedSearchCV)",
                "max_depth": "8, 12, atau None (tuned)",
                "random_state": "42",
                "n_jobs": "-1 (paralel)",
            },
            "pros": [
                "Robust terhadap overfitting",
                "Tidak perlu scaling data",
                "Dapat menghitung feature importance",
                "Performa sangat baik (R² > 0.998)",
            ],
            "cons": [
                "Model berukuran besar (~50MB)",
                "Tidak menangkap pola sekuensial/temporal secara native",
                "Membutuhkan feature engineering manual",
            ],
        },
        {
            "name": "Gradient Boosting",
            "type": "Ensemble (Boosting)",
            "color": "#f97316",
            "bg": "rgba(249, 115, 22, 0.08)",
            "border": "rgba(249, 115, 22, 0.25)",
            "description": (
                "Ensemble dari Decision Tree yang dilatih secara sekuensial. "
                "Setiap tree baru belajar dari kesalahan (residual) tree sebelumnya, "
                "sehingga model terus membaik. Dikenal sebagai salah satu metode terbaik untuk data tabular."
            ),
            "parameters": {
                "n_estimators": "120-360 (tuned via RandomizedSearchCV)",
                "learning_rate": "0.05-0.20 (tuned)",
                "max_depth": "3",
                "random_state": "42",
            },
            "pros": [
                "Performa terbaik di antara ketiga model (R² = 0.9989)",
                "Model ringan (~330KB)",
                "Generalisasi yang baik dengan learning rate rendah",
                "Feature importance yang interpretable",
            ],
            "cons": [
                "Training lebih lambat dari Random Forest",
                "Sensitif terhadap hyperparameter",
                "Sequential training (tidak bisa paralel antar tree)",
            ],
        },
        {
            "name": "LSTM",
            "type": "Deep Learning (RNN)",
            "color": "#ef4444",
            "bg": "rgba(239, 68, 68, 0.08)",
            "border": "rgba(239, 68, 68, 0.25)",
            "description": (
                "Long Short-Term Memory, arsitektur Recurrent Neural Network yang mampu mempelajari "
                "dependensi jangka panjang dalam data sekuensial. Menggunakan mekanisme gate "
                "(forget, input, output) untuk mengontrol aliran informasi."
            ),
            "parameters": {
                "architecture": "LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16, ReLU) → Dense(1)",
                "optimizer": "Adam (learning rate = 0.001)",
                "loss_function": "Mean Squared Error (MSE)",
                "window_size": "24 jam",
                "batch_size": "64",
                "epochs": "50 (Early Stopping, patience=10)",
            },
            "pros": [
                "Dapat mempelajari pola temporal/sekuensial secara otomatis",
                "Tidak perlu feature engineering manual",
                "Skalabel untuk data streaming/real-time",
                "Model sangat ringan (~400KB)",
            ],
            "cons": [
                "Performa lebih rendah (R² = 0.52) hanya dengan 1 fitur",
                "Butuh GPU untuk training cepat",
                "Memerlukan normalisasi data (StandardScaler)",
                "Hyperparameter tuning lebih kompleks",
            ],
        },
    ]

    for model in models_info:
        with st.expander(f"{'🌲' if model['name'] == 'Random Forest' else '📈' if model['name'] == 'Gradient Boosting' else '🧠'} {model['name']} — {model['type']}", expanded=False):
            st.markdown(f"**{model['description']}**")

            st.markdown("---")
            st.markdown("**Parameter & Konfigurasi:**")
            for key, val in model["parameters"].items():
                st.markdown(f"- `{key}`: {val}")

            col_pro, col_con = st.columns(2)
            with col_pro:
                st.markdown("**Kelebihan:**")
                for p in model["pros"]:
                    st.markdown(f"- ✅ {p}")
            with col_con:
                st.markdown("**Kekurangan:**")
                for c in model["cons"]:
                    st.markdown(f"- ❌ {c}")

    # Feature Engineering
    st.markdown("---")
    st.subheader("Rekayasa Fitur (Feature Engineering)")

    st.markdown("**Fitur untuk Random Forest & Gradient Boosting (24 fitur):**")

    fe_col1, fe_col2 = st.columns(2)
    with fe_col1:
        st.markdown("""
        **Fitur Sensor (6):**
        `Global_reactive_power`, `Voltage`, `Global_intensity`,
        `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`

        **Fitur Kalender (4):**
        `hour`, `dayofweek`, `is_weekend`, `month`
        """)
    with fe_col2:
        st.markdown("""
        **Fitur Lag (6):**
        `lag1`, `lag2`, `lag3`, `lag6`, `lag12`, `lag24`

        **Fitur Rolling (8):**
        `rollmean3/6/12/24`, `rollstd3/6/12/24`
        """)

    st.markdown("**Input untuk LSTM:**")
    st.code("Window 24 jam Global_active_power (StandardScaler normalized) - Shape: (1, 24, 1)")

    # Dataset Info
    st.markdown("---")
    st.subheader("Tentang Dataset")

    st.markdown(
        f'Sumber: <a href="{UCI_DATASET_URL}" target="_blank" class="source-link">'
        f'UCI Machine Learning Repository - Individual Household Electric Power Consumption</a>',
        unsafe_allow_html=True,
    )

    ds_info = {
        "Periode": "Desember 2006 - November 2010 (~4 tahun)",
        "Resolusi Asli": "Per menit (2.075.259 records)",
        "Resolusi Digunakan": "Per jam / 1H (resampled mean) → 34.168 records",
        "Target": "Global_active_power (kW)",
        "Split Data": "Train 70% (23.902) | Validation 15% (5.121) | Test 15% (5.121)",
        "Metode Split": "Time-based split (kronologis, tanpa shuffling)",
    }

    for key, val in ds_info.items():
        st.markdown(f"- **{key}:** {val}")

    st.markdown(
        f'\n📎 [Unduh dataset dari UCI Repository]({UCI_DATASET_URL})',
    )
