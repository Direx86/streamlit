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
import plotly.graph_objects as go
from pathlib import Path

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

RF_MODEL_PATH   = MODELS_DIR / "random_forest_regressor_TUNED.joblib"
GB_MODEL_PATH   = MODELS_DIR / "gradient_boosting_regressor_TUNED.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
METRICS_CSV_PATH = MODELS_DIR / "test_metrics_default.csv"

SAMPLE_DATA_PATH    = DATA_DIR / "sample_data.json"
FULL_TEST_DATA_PATH = DATA_DIR / "full_test_data.json"

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"

# ─── GLOBAL CSS — dark theme only ────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Headings / text ── */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    p, li, label, span {
        color: #cbd5e1;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] thead tr th {
        background: #0f172a !important;
        color: #94a3b8 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stDataFrame"] tbody tr td {
        color: #e2e8f0 !important;
        font-size: 13px !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #1e293b;
        border: 1px solid #334155 !important;
        border-radius: 14px;
    }
    [data-testid="stExpander"] summary {
        color: #f1f5f9 !important;
        font-weight: 600;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* ── Radio & checkbox ── */
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: #cbd5e1 !important;
    }

    /* ── Hide code blocks — never show raw code ── */
    code, pre code {
        background: #1e293b !important;
        color: #7dd3fc !important;
        border-radius: 6px;
        padding: 2px 6px;
    }
    [data-testid="stCodeBlock"] {
        display: none !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #334155 !important;
    }

    /* ── Caption / small text ── */
    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 12px;
    }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly .bg {
        fill: #1e293b !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1e293b; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #64748b; }

    /* ── Containers & borders ── */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1e293b;
        border-color: #334155 !important;
        border-radius: 14px;
    }

    /* ── Info box ── */
    [data-testid="stAlert"] {
        background: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 12px;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] a {
        color: #e2e8f0 !important;
    }

    /* ── Variable badge ── */
    .var-badge {
        background: #0f172a;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        color: #7dd3fc;
        margin: 6px 0;
    }
    .var-badge .unit {
        color: #94a3b8;
        font-weight: 400;
    }

    /* ── Feature tag list ── */
    .feature-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 4px 0;
    }
    .feature-tag {
        background: #0f172a;
        color: #7dd3fc;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid #334155;
    }

    /* ── LSTM info box ── */
    .lstm-info {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 3px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.7;
    }
    .lstm-info strong { color: #f1f5f9; }
    .lstm-info .val { color: #7dd3fc; font-weight: 500; }

    /* ── Utility badges ── */
    .badge-rf   { display:inline-block; padding:3px 10px; border-radius:8px; font-weight:700; font-size:12px; background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); }
    .badge-gb   { display:inline-block; padding:3px 10px; border-radius:8px; font-weight:700; font-size:12px; background:rgba(249,115,22,0.15); color:#fb923c; border:1px solid rgba(249,115,22,0.3); }
    .badge-lstm { display:inline-block; padding:3px 10px; border-radius:8px; font-weight:700; font-size:12px; background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.3); }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ────────────────────────────────────────────
@st.cache_data
def extract_dataset():
    if not DATASET_TXT_PATH.exists() and DATASET_ZIP_PATH.exists():
        with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as z:
            z.extract("household_power_consumption.txt", MODELS_DIR)
    return DATASET_TXT_PATH.exists()


@st.cache_data
def load_sample_data():
    if SAMPLE_DATA_PATH.exists():
        with open(SAMPLE_DATA_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_full_test_data():
    if FULL_TEST_DATA_PATH.exists():
        with open(FULL_TEST_DATA_PATH) as f:
            return json.load(f)
    return None


def compute_metrics(data_points):
    """Compute MAE, RMSE, R² from a list of data-point dicts."""
    models_cfg = [
        {"key": "rf_pred",   "name": "Random Forest"},
        {"key": "gb_pred",   "name": "Gradient Boosting"},
        {"key": "lstm_pred", "name": "LSTM"},
    ]
    results = []
    for m in models_cfg:
        valid = [d for d in data_points if d.get(m["key"]) and d[m["key"]] != 0]
        if not valid:
            results.append({"model": m["name"], "MAE": 0.0, "RMSE": 0.0, "R2": 0.0})
            continue
        actual = np.array([d["actual"]   for d in valid])
        pred   = np.array([d[m["key"]]  for d in valid])
        errors = actual - pred
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        results.append({"model": m["name"], "MAE": mae, "RMSE": rmse, "R2": r2})
    return results


# ─── CHART ───────────────────────────────────────────────────
def create_prediction_chart(data, title, show_models=None, height=500):
    if not data:
        return None
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if show_models is None:
        show_models = {"actual": True, "rf": True, "gb": True, "lstm": True}

    fig = go.Figure()
    if show_models.get("actual") and "actual" in df.columns:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["actual"],
            name="Data Aktual", mode="lines",
            line=dict(color="#3b82f6", width=2.5)))
    if show_models.get("rf") and "rf_pred" in df.columns:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["rf_pred"],
            name="Random Forest", mode="lines",
            line=dict(color="#10b981", width=2, dash="dash")))
    if show_models.get("gb") and "gb_pred" in df.columns:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["gb_pred"],
            name="Gradient Boosting", mode="lines",
            line=dict(color="#f97316", width=2, dash="dot")))
    if show_models.get("lstm") and "lstm_pred" in df.columns:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["lstm_pred"],
            name="LSTM", mode="lines",
            line=dict(color="#ef4444", width=2, dash="dashdot")))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
        xaxis=dict(title="Waktu", gridcolor="#334155", color="#94a3b8",
                   tickfont=dict(size=11), showline=True, linecolor="#475569"),
        yaxis=dict(title="Global Active Power (kW)", gridcolor="#334155",
                   color="#94a3b8", tickfont=dict(size=11),
                   showline=True, linecolor="#475569"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#475569",
                    borderwidth=1, font=dict(size=12)),
        height=height,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


# ─── METRICS TABLE ───────────────────────────────────────────
def render_metrics_table(metrics, title, subtitle=None):
    """Render a styled metrics table using st.dataframe."""
    if not metrics:
        return

    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

    df = pd.DataFrame(metrics)
    df.columns = ["Model", "MAE", "RMSE", "R²"]

    best_mae_idx  = int(df["MAE"].idxmin())
    best_rmse_idx = int(df["RMSE"].idxmin())
    best_r2_idx   = int(df["R²"].idxmax())

    # Build display dataframe with formatted strings + star for best
    display = pd.DataFrame()
    display["Model"] = df["Model"]
    display["MAE ↓"]  = [
        f"{v:.4f} ⭐" if i == best_mae_idx  else f"{v:.4f}"
        for i, v in enumerate(df["MAE"])
    ]
    display["RMSE ↓"] = [
        f"{v:.4f} ⭐" if i == best_rmse_idx else f"{v:.4f}"
        for i, v in enumerate(df["RMSE"])
    ]
    display["R² ↑"]   = [
        f"{v:.4f} ⭐" if i == best_r2_idx   else f"{v:.4f}"
        for i, v in enumerate(df["R²"])
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model":  st.column_config.TextColumn("Model",  width="medium"),
            "MAE ↓":  st.column_config.TextColumn("MAE ↓",  width="small"),
            "RMSE ↓": st.column_config.TextColumn("RMSE ↓", width="small"),
            "R² ↑":   st.column_config.TextColumn("R² ↑",   width="small"),
        },
    )


# ─── SIDEBAR ─────────────────────────────────────────────────
extract_dataset()

with st.sidebar:
    st.markdown("## ⚡ Smart Home Energy")
    st.markdown("**Prediksi Konsumsi Energi Rumah Tangga**")
    st.divider()

    page = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "📊 Prediksi & Perbandingan", "🧠 Informasi Model"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Dataset")
    st.markdown(
        f"[📂 UCI Power Consumption]({UCI_DATASET_URL})",
    )
    st.caption("Desember 2006 – November 2010")
    st.divider()
    st.caption("Skripsi — Franscen Yosafat Sinambela · 2025")
    st.caption("RF · GB · LSTM Comparison")


# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard Smart Home")
    st.caption("Monitoring & Prediksi Konsumsi Energi Rumah Tangga")

    sample_data = load_sample_data()

    if sample_data and sample_data.get("data"):
        data = sample_data["data"]
        latest  = data[-1]
        avg_val = sum(d["actual"] for d in data) / len(data)
        max_val = max(d["actual"] for d in data)
        min_val = min(d["actual"] for d in data)

        # ── Stat Cards ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("⚡ Konsumsi Terakhir",  f"{latest['actual']:.2f} kW",
                      help="Global Active Power (nilai terbaru dalam test set)")
        with c2:
            st.metric("📈 Rata-rata 7 Hari",   f"{avg_val:.2f} kW",
                      help=f"{len(data)} data point")
        with c3:
            st.metric("🔺 Konsumsi Tertinggi", f"{max_val:.2f} kW",
                      help="Puncak pemakaian dalam 7 hari")
        with c4:
            st.metric("🔻 Konsumsi Terendah",  f"{min_val:.2f} kW",
                      help="Pemakaian minimum dalam 7 hari")

        st.divider()

        # ── Dataset Variables ──
        st.subheader("📋 Variabel Dataset UCI")
        st.caption(
            f"Sumber: [UCI Machine Learning Repository]({UCI_DATASET_URL}) "
            "— Individual Household Electric Power Consumption"
        )

        vars_data = [
            ("⚡ Global Active Power",   "Global_active_power",   "kW",
             "Total daya aktif rumah tangga — **TARGET prediksi**",   "#3b82f6"),
            ("🔄 Global Reactive Power", "Global_reactive_power",  "kW",
             "Total daya reaktif rumah tangga",                        "#8b5cf6"),
            ("⚡ Voltage",               "Voltage",                "Volt",
             "Tegangan listrik rata-rata per menit",                   "#eab308"),
            ("🔌 Global Intensity",      "Global_intensity",       "Ampere",
             "Arus listrik rata-rata per menit",                       "#f97316"),
            ("🍳 Sub Metering 1",        "Sub_metering_1",         "Wh",
             "Dapur: dishwasher, oven, microwave",                     "#10b981"),
            ("👕 Sub Metering 2",        "Sub_metering_2",         "Wh",
             "Laundry: mesin cuci, pengering, kulkas",                 "#06b6d4"),
            ("🌡️ Sub Metering 3",       "Sub_metering_3",         "Wh",
             "Pemanas air listrik & AC",                               "#ef4444"),
        ]

        cols = st.columns(4)
        for i, (name, var, unit, desc, color) in enumerate(vars_data):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.markdown(
                        f'<div class="var-badge" style="border-left:3px solid {color}">'
                        f'{var} <span class="unit">({unit})</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(desc)

        st.divider()

        # ── Data Source Badge ──
        st.info(
            f"📊 **Data Asli:** [UCI Household Power Consumption]({UCI_DATASET_URL})  "
            f"— Test Set: **{len(data)} points**",
            icon=None,
        )

        # ── Chart ──
        st.subheader("📉 Grafik Perbandingan Prediksi 3 Model")
        fig = create_prediction_chart(
            data,
            f"Perbandingan Prediksi 3 Model ({len(data)} Jam)",
            height=460,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # ── Metrics ──
        if sample_data.get("metrics"):
            st.divider()
            render_metrics_table(
                sample_data["metrics"],
                "🏆 Perbandingan Metrik Evaluasi (Test Set)",
                "Semakin kecil MAE & RMSE = semakin baik.  "
                "Semakin besar R² (mendekati 1) = semakin baik.  ⭐ = terbaik.",
            )
    else:
        st.warning(
            "Data sampel tidak ditemukan. "
            "Jalankan `python backend/generate_sample_data.py` terlebih dahulu.",
            icon="⚠️",
        )


# ═══════════════════════════════════════════════════════════
# PAGE: PREDIKSI & PERBANDINGAN
# ═══════════════════════════════════════════════════════════
elif page == "📊 Prediksi & Perbandingan":
    st.title("📊 Prediksi & Perbandingan")
    st.caption(
        f"Bandingkan hasil prediksi RF, GB, dan LSTM pada "
        f"[dataset UCI asli]({UCI_DATASET_URL})"
    )

    sample_data   = load_sample_data()
    full_test_data = load_full_test_data()

    total_pts = (
        len(full_test_data["data"]) if full_test_data
        else (len(sample_data["data"]) if sample_data else 0)
    )
    st.info(
        f"📊 **Data Asli:** [UCI Household Power Consumption]({UCI_DATASET_URL})  "
        f"— Test Set: **{total_pts} points**",
        icon=None,
    )

    st.divider()

    # ── Controls ──
    st.subheader("⚙️ Kontrol Visualisasi")
    col_range, col_models = st.columns([1, 1])

    with col_range:
        st.markdown("**Rentang Waktu**")
        time_options = {
            "24 Jam": 24, "48 Jam": 48, "7 Hari": 168,
            "30 Hari": 720, "Full Test Set": 0,
        }
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
        with mc1: show_actual = st.checkbox("Aktual", value=True)
        with mc2: show_rf     = st.checkbox("RF",     value=True)
        with mc3: show_gb     = st.checkbox("GB",     value=True)
        with mc4: show_lstm   = st.checkbox("LSTM",   value=True)

    show_models = {
        "actual": show_actual, "rf": show_rf,
        "gb": show_gb, "lstm": show_lstm,
    }

    # ── Pick data source ──
    source = full_test_data if (time_range > 168 or time_range == 0) else sample_data

    if source and source.get("data"):
        data = source["data"]
        if time_range > 0:
            data = data[-time_range:]

        # ── Main Chart ──
        fig = create_prediction_chart(
            data,
            f"Grafik Perbandingan — {time_label} ({len(data)} points)",
            show_models=show_models,
            height=500,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # ── Error Analysis ──
        with st.expander(f"📊 Analisis Error — {len(data)} points", expanded=False):
            models_analysis = [
                {"key": "rf_pred",   "name": "Random Forest",    "color": "green"},
                {"key": "gb_pred",   "name": "Gradient Boosting","color": "orange"},
                {"key": "lstm_pred", "name": "LSTM",             "color": "red"},
            ]
            ecols = st.columns(3)
            for i, m in enumerate(models_analysis):
                valid = [d for d in data if d.get(m["key"]) and d[m["key"]] != 0]
                if valid:
                    errors = [abs(d["actual"] - d[m["key"]]) for d in valid]
                    mae     = sum(errors) / len(errors)
                    rmse    = math.sqrt(sum(e**2 for e in errors) / len(errors))
                    max_err = max(errors)
                    with ecols[i]:
                        with st.container(border=True):
                            st.markdown(f"**:{m['color']}[{m['name']}]**")
                            st.metric("MAE",       f"{mae:.4f} kW")
                            st.metric("RMSE",      f"{rmse:.4f} kW")
                            st.metric("Max Error", f"{max_err:.4f} kW")
                            st.caption(f"{len(valid)} data points")

        # ── Dynamic Metrics ──
        dynamic_metrics = compute_metrics(data)
        render_metrics_table(
            dynamic_metrics,
            f"📈 Metrik Evaluasi — {time_label} ({len(data)} points)",
            "Dihitung ulang secara dinamis berdasarkan rentang waktu yang dipilih.  ⭐ = terbaik.",
        )

        st.markdown("")

        # ── Official Metrics ──
        if sample_data and sample_data.get("metrics"):
            render_metrics_table(
                sample_data["metrics"],
                "🏆 Metrik Evaluasi Resmi — Full Test Set (5.121 points)",
                "Metrik tetap dari evaluasi seluruh test set (15% data).  "
                "Ini adalah angka yang dilaporkan di skripsi.  ⭐ = terbaik.",
            )

        st.divider()

        # ── Penjelasan Metrik ──
        st.subheader("📖 Penjelasan Metrik Evaluasi")

        with st.container(border=True):
            st.markdown("#### Apa itu MAE, RMSE, dan R²?")
            st.markdown("")

            tab_mae, tab_rmse, tab_r2 = st.tabs(["MAE", "RMSE", "R²"])

            with tab_mae:
                st.markdown("##### 1. MAE — Mean Absolute Error (Rata-rata Kesalahan Absolut)")
                st.markdown("""
MAE mengukur **rata-rata selisih absolut** antara nilai prediksi dan nilai aktual.

**Rumus:**  `MAE = (1/n) × Σ|yᵢ − ŷᵢ|`

**Interpretasi:**
- MAE menunjukkan seberapa besar rata-rata kesalahan prediksi model **dalam satuan asli (kW)**.
- Semakin kecil MAE → semakin akurat model.
- Contoh: `MAE = 0.0158 kW` artinya rata-rata prediksi meleset hanya **~15.8 Watt** dari nilai sebenarnya.

**Kelebihan MAE:**
- Mudah diinterpretasikan — satuannya sama dengan data asli.
- Tidak terlalu sensitif terhadap outlier (pencilan data).
                """)

            with tab_rmse:
                st.markdown("##### 2. RMSE — Root Mean Squared Error (Akar Rata-rata Kesalahan Kuadrat)")
                st.markdown("""
RMSE mengukur **akar dari rata-rata kuadrat selisih** antara prediksi dan nilai aktual.

**Rumus:**  `RMSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]`

**Interpretasi:**
- RMSE juga dalam satuan kW, namun memberikan **penalti lebih besar untuk kesalahan yang besar**.
- Jika `RMSE ≫ MAE` → ada beberapa prediksi yang meleset jauh (outlier error).
- Jika `RMSE ≈ MAE` → prediksi konsisten tanpa kesalahan ekstrem ✅

**Perbandingan MAE vs RMSE:**
Pada RF dan GB, RMSE ≈ MAE → model sangat konsisten, nyaris tidak ada kesalahan besar.
                """)

            with tab_r2:
                st.markdown("##### 3. R² — R-Squared (Koefisien Determinasi)")
                st.markdown("""
R² mengukur **proporsi variabilitas data yang dapat dijelaskan oleh model**.

**Rumus:**  `R² = 1 − (SS_res / SS_tot)`
- `SS_res = Σ(yᵢ − ŷᵢ)²`  (total kesalahan model)
- `SS_tot = Σ(yᵢ − ȳ)²`   (total variasi data)

**Interpretasi nilai R²:**

| Nilai R² | Arti |
|----------|------|
| `1.0000` | Model sempurna — prediksi 100% cocok |
| `0.9989` | Model menjelaskan **99.89%** variasi data — sangat akurat ✅ |
| `0.5243` | Model menjelaskan **52.43%** variasi data — cukup, banyak variasi tidak tertangkap |
| `0.0`    | Tidak lebih baik dari prediksi rata-rata |
| `< 0`   | Lebih buruk dari prediksi rata-rata |

**Kelebihan R²:**
- Memberikan gambaran menyeluruh tentang kualitas model.
- Tidak bergantung pada skala/satuan data.
                """)

        st.divider()

        # ── Interpretasi Hasil ──
        st.subheader("🔍 Interpretasi Hasil")

        official = sample_data["metrics"] if sample_data and sample_data.get("metrics") else []
        rf_m   = next((m for m in official if m["model"] == "RandomForest"),      {})
        gb_m   = next((m for m in official if m["model"] == "GradientBoosting"),  {})
        lstm_m = next((m for m in official if m["model"] == "LSTM"),              {})

        # RF card
        with st.container(border=True):
            st.markdown("#### 🌲 Random Forest (RF)")
            r1, r2, r3 = st.columns(3)
            with r1: st.metric("MAE",  f"{rf_m.get('MAE', 0.0166):.4f} kW")
            with r2: st.metric("RMSE", f"{rf_m.get('RMSE', 0.0249):.4f} kW")
            with r3: st.metric("R²",   f"{rf_m.get('R2', 0.9988):.4f}")
            st.markdown(f"""
- **MAE sangat kecil** (~{rf_m.get('MAE', 0.0166):.3f} kW) → prediksi hampir selalu sangat dekat dengan nilai aktual.
- **RMSE ≈ MAE** → tidak ada kesalahan ekstrem, prediksi sangat konsisten.
- **R² = {rf_m.get('R2', 0.9988):.4f}** → model menjelaskan **{rf_m.get('R2', 0.9988)*100:.2f}%** variasi konsumsi energi.
- Keunggulan berkat **24 fitur** (sensor + kalender + lag + rolling statistics) yang memberikan konteks historis yang kaya.
            """)

        # GB card
        with st.container(border=True):
            st.markdown("#### 📈 Gradient Boosting (GB) — 🥇 Model Terbaik")
            g1, g2, g3 = st.columns(3)
            with g1: st.metric("MAE",  f"{gb_m.get('MAE', 0.0158):.4f} kW",  delta="Best")
            with g2: st.metric("RMSE", f"{gb_m.get('RMSE', 0.0239):.4f} kW", delta="Best")
            with g3: st.metric("R²",   f"{gb_m.get('R2', 0.9989):.4f}",       delta="Best")
            st.markdown(f"""
- **MAE terendah** ({gb_m.get('MAE', 0.0158):.4f} kW) → akurasi tertinggi di antara ketiga model.
- **RMSE terendah** ({gb_m.get('RMSE', 0.0239):.4f} kW) → paling konsisten, kesalahan besar paling jarang.
- **R² tertinggi** ({gb_m.get('R2', 0.9989):.4f}) → menjelaskan **{gb_m.get('R2', 0.9989)*100:.2f}%** variasi data.
- GB menggunakan strategi *sequential learning* (boosting): setiap tree baru secara eksplisit **mengoreksi kesalahan tree sebelumnya**.
- Ukuran model hanya ~330 KB (vs RF ~48 MB) — **lebih efisien untuk deployment**.
            """)

        # LSTM card
        with st.container(border=True):
            st.markdown("#### 🧠 LSTM (Long Short-Term Memory)")
            l1, l2, l3 = st.columns(3)
            with l1: st.metric("MAE",  f"{lstm_m.get('MAE', 0.3462):.4f} kW")
            with l2: st.metric("RMSE", f"{lstm_m.get('RMSE', 0.4896):.4f} kW")
            with l3: st.metric("R²",   f"{lstm_m.get('R2', 0.5243):.4f}")
            st.markdown(f"""
- **MAE ~{lstm_m.get('MAE', 0.3462):.2f} kW** → sekitar **{lstm_m.get('MAE', 0.3462)/gb_m.get('MAE', 0.0158):.0f}× lebih besar** dari Gradient Boosting.
- **RMSE ≫ MAE** → ada beberapa prediksi yang meleset cukup jauh dari nilai aktual.
- **R² = {lstm_m.get('R2', 0.5243):.4f}** → hanya menjelaskan **{lstm_m.get('R2', 0.5243)*100:.2f}%** variasi data.
- **Penyebab performa lebih rendah:** LSTM hanya menggunakan **1 fitur** (Global_active_power),
  tanpa sensor lain, informasi waktu, atau fitur lag/rolling seperti RF & GB (24 fitur).
- LSTM tetap mampu menangkap **tren umum** konsumsi energi harian.
- Berfungsi sebagai **baseline univariate** untuk menunjukkan pentingnya feature engineering.
            """)

        st.divider()

        # ── Kesimpulan ──
        with st.container(border=True):
            st.markdown("#### 🏆 Perbandingan RF vs GB: Mana yang Terbaik?")

            comp_df = pd.DataFrame({
                "Metrik":           ["MAE (↓ lebih baik)", "RMSE (↓ lebih baik)", "R² (↑ lebih baik)"],
                "Random Forest":    [f"{rf_m.get('MAE', 0.0166):.4f}",
                                     f"{rf_m.get('RMSE', 0.0249):.4f}",
                                     f"{rf_m.get('R2', 0.9988):.4f}"],
                "Gradient Boosting":[f"{gb_m.get('MAE', 0.0158):.4f} ⭐",
                                     f"{gb_m.get('RMSE', 0.0239):.4f} ⭐",
                                     f"{gb_m.get('R2', 0.9989):.4f} ⭐"],
                "Pemenang":         ["Gradient Boosting", "Gradient Boosting", "Gradient Boosting"],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("""
**Kesimpulan: Gradient Boosting adalah Model Terbaik**

GB mengungguli semua model di ketiga metrik:

1. **MAE terendah** → rata-rata error ~5% lebih kecil dari RF.
2. **RMSE terendah** → lebih konsisten, kesalahan besar lebih jarang (~4% lebih kecil dari RF).
3. **R² tertinggi** → menjelaskan variasi data sedikit lebih baik dari RF.
4. **Ukuran model lebih kecil** (~330 KB vs ~48 MB) → lebih efisien untuk deployment.

Metode *boosting* (sequential learning) secara iteratif mengoreksi kesalahan tree sebelumnya,
menghasilkan akurasi lebih tinggi dibanding *bagging* (RF) pada dataset ini.
            """)

        st.caption(
            f"Semua data berasal dari **[dataset UCI asli]({UCI_DATASET_URL})** "
            "(household_power_consumption.txt). Periode test set: April – November 2010."
        )
    else:
        st.warning("Data tidak ditemukan. Pastikan file JSON sudah di-generate.", icon="⚠️")


# ═══════════════════════════════════════════════════════════
# PAGE: INFORMASI MODEL
# ═══════════════════════════════════════════════════════════
elif page == "🧠 Informasi Model":
    st.title("🧠 Informasi Model")
    st.caption("Detail arsitektur, parameter, dan karakteristik masing-masing model prediksi")

    models_info = [
        {
            "icon": "🌲",
            "name": "Random Forest",
            "type": "Ensemble (Bagging)",
            "description": (
                "Ensemble dari banyak Decision Tree yang dilatih secara **paralel** dengan teknik bagging. "
                "Setiap tree dilatih pada subset data acak, lalu hasil prediksi dirata-ratakan. "
                "Robust terhadap overfitting dan mampu menangkap hubungan non-linear."
            ),
            "parameters": {
                "n_estimators":  "80–200 (tuned via RandomizedSearchCV)",
                "max_depth":     "8, 12, atau None (tuned)",
                "random_state":  "42",
                "n_jobs":        "-1 (paralel)",
            },
            "pros": [
                "Robust terhadap overfitting",
                "Tidak perlu scaling data",
                "Dapat menghitung feature importance",
                "Performa sangat baik (R² > 0.998)",
            ],
            "cons": [
                "Model berukuran besar (~48 MB)",
                "Tidak menangkap pola sekuensial secara native",
                "Membutuhkan feature engineering manual",
            ],
        },
        {
            "icon": "📈",
            "name": "Gradient Boosting",
            "type": "Ensemble (Boosting)",
            "description": (
                "Ensemble dari Decision Tree yang dilatih secara **sekuensial**. "
                "Setiap tree baru belajar dari kesalahan (residual) tree sebelumnya, "
                "sehingga model terus membaik. Dikenal sebagai salah satu metode terbaik untuk data tabular."
            ),
            "parameters": {
                "n_estimators":  "120–360 (tuned via RandomizedSearchCV)",
                "learning_rate": "0.05–0.20 (tuned)",
                "max_depth":     "3",
                "random_state":  "42",
            },
            "pros": [
                "Performa terbaik di antara ketiga model (R² = 0.9989)",
                "Model ringan (~330 KB)",
                "Generalisasi baik dengan learning rate rendah",
                "Feature importance yang interpretable",
            ],
            "cons": [
                "Training lebih lambat dari Random Forest",
                "Sensitif terhadap hyperparameter",
                "Sequential training (tidak bisa paralel antar tree)",
            ],
        },
        {
            "icon": "🧠",
            "name": "LSTM",
            "type": "Deep Learning (RNN)",
            "description": (
                "Long Short-Term Memory — arsitektur Recurrent Neural Network yang mampu mempelajari "
                "**dependensi jangka panjang** dalam data sekuensial. Menggunakan mekanisme gate "
                "(forget, input, output) untuk mengontrol aliran informasi."
            ),
            "parameters": {
                "architecture":  "LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16, ReLU) → Dense(1)",
                "optimizer":     "Adam (learning rate = 0.001)",
                "loss_function": "Mean Squared Error (MSE)",
                "window_size":   "24 jam",
                "batch_size":    "64",
                "epochs":        "50 (Early Stopping, patience=10)",
            },
            "pros": [
                "Dapat mempelajari pola temporal/sekuensial secara otomatis",
                "Tidak perlu feature engineering manual",
                "Skalabel untuk data streaming/real-time",
                "Model sangat ringan (~400 KB)",
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
        with st.expander(
            f"{model['icon']} {model['name']} — {model['type']}",
            expanded=False,
        ):
            st.markdown(model["description"])
            st.divider()

            st.markdown("**⚙️ Parameter & Konfigurasi**")
            param_df = pd.DataFrame(
                list(model["parameters"].items()),
                columns=["Parameter", "Nilai"],
            )
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            col_pro, col_con = st.columns(2)
            with col_pro:
                st.markdown("**✅ Kelebihan**")
                for p in model["pros"]:
                    st.markdown(f"- {p}")
            with col_con:
                st.markdown("**❌ Kekurangan**")
                for c in model["cons"]:
                    st.markdown(f"- {c}")

    st.divider()

    # ── Feature Engineering ──
    st.subheader("🛠️ Rekayasa Fitur (Feature Engineering)")
    st.markdown("**Fitur untuk Random Forest & Gradient Boosting (24 fitur total):**")

    fe_col1, fe_col2 = st.columns(2)
    with fe_col1:
        with st.container(border=True):
            st.markdown("**Fitur Sensor (6)**")
            st.markdown(
                '<div class="feature-tags">'
                '<span class="feature-tag">Global_reactive_power</span>'
                '<span class="feature-tag">Voltage</span>'
                '<span class="feature-tag">Global_intensity</span>'
                '<span class="feature-tag">Sub_metering_1</span>'
                '<span class="feature-tag">Sub_metering_2</span>'
                '<span class="feature-tag">Sub_metering_3</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.markdown("**Fitur Kalender (4)**")
            st.markdown(
                '<div class="feature-tags">'
                '<span class="feature-tag">hour</span>'
                '<span class="feature-tag">dayofweek</span>'
                '<span class="feature-tag">is_weekend</span>'
                '<span class="feature-tag">month</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    with fe_col2:
        with st.container(border=True):
            st.markdown("**Fitur Lag (6)**")
            st.markdown(
                '<div class="feature-tags">'
                '<span class="feature-tag">lag1</span>'
                '<span class="feature-tag">lag2</span>'
                '<span class="feature-tag">lag3</span>'
                '<span class="feature-tag">lag6</span>'
                '<span class="feature-tag">lag12</span>'
                '<span class="feature-tag">lag24</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.markdown("**Fitur Rolling (8)**")
            st.markdown(
                '<div class="feature-tags">'
                '<span class="feature-tag">rollmean3</span>'
                '<span class="feature-tag">rollmean6</span>'
                '<span class="feature-tag">rollmean12</span>'
                '<span class="feature-tag">rollmean24</span>'
                '<span class="feature-tag">rollstd3</span>'
                '<span class="feature-tag">rollstd6</span>'
                '<span class="feature-tag">rollstd12</span>'
                '<span class="feature-tag">rollstd24</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("**Input untuk LSTM:**")
    st.markdown(
        '<div class="lstm-info">'
        '<strong>Window:</strong> <span class="val">24 jam</span> · '
        '<strong>Fitur:</strong> <span class="val">Global_active_power</span> · '
        '<strong>Normalisasi:</strong> <span class="val">StandardScaler</span><br>'
        '<strong>Shape:</strong> <span class="val">(1, 24, 1)</span> · '
        '<strong>Mean:</strong> <span class="val">1.086397</span> · '
        '<strong>Std:</strong> <span class="val">0.929282</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Dataset Info ──
    st.subheader("📂 Tentang Dataset")
    st.markdown(
        f"**Sumber:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption]({UCI_DATASET_URL})"
    )

    ds_info = {
        "Periode":          "Desember 2006 – November 2010 (~4 tahun)",
        "Resolusi Asli":    "Per menit (2.075.259 records)",
        "Resolusi Digunakan": "Per jam / 1H (resampled mean) → 34.168 records",
        "Target":           "Global_active_power (kW)",
        "Split Data":       "Train 70% (23.902) | Validation 15% (5.121) | Test 15% (5.121)",
        "Metode Split":     "Time-based split (kronologis, tanpa shuffling)",
    }
    ds_df = pd.DataFrame(list(ds_info.items()), columns=["Detail", "Keterangan"])
    st.dataframe(ds_df, use_container_width=True, hide_index=True)

    st.markdown(f"📎 [Unduh dataset dari UCI Repository]({UCI_DATASET_URL})")
