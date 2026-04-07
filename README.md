# Smart Home Energy Prediction

Prediksi Konsumsi Energi Rumah Tangga menggunakan **Random Forest**, **Gradient Boosting**, dan **LSTM** pada Sistem Smart Home.

Aplikasi web **Streamlit** berbasis dataset [UCI - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

> **Arsitektur:** Single repository Streamlit (frontend & backend dalam satu aplikasi Python). Tidak memerlukan Vercel atau Railway.

## Dataset

**Sumber:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

| Detail | Keterangan |
|---|---|
| File | `household_power_consumption.txt` (dari ZIP dalam repo) |
| Jumlah Baris | 2.075.259 (per menit) |
| Periode | Desember 2006 - November 2010 |
| Resampling | Rata-rata per jam (hourly) → 34.168 baris |
| Split Data | Train 70% / Val 15% / Test 15% |

### Variabel Dataset

| Variabel | Unit | Deskripsi |
|---|---|---|
| Global_active_power | kW | Total daya aktif rumah tangga (**TARGET**) |
| Global_reactive_power | kW | Total daya reaktif rumah tangga |
| Voltage | Volt | Tegangan listrik rata-rata |
| Global_intensity | Ampere | Arus listrik rata-rata |
| Sub_metering_1 | Wh | Dapur: dishwasher, oven, microwave |
| Sub_metering_2 | Wh | Laundry: mesin cuci, pengering, kulkas |
| Sub_metering_3 | Wh | Pemanas air listrik & AC |

## Model Machine Learning

| Model | Tipe | Input | File |
|---|---|---|---|
| Random Forest (TUNED) | Ensemble (Bagging) | 24 fitur | `random_forest_regressor_TUNED.joblib` (~48 MB) |
| Gradient Boosting (TUNED) | Ensemble (Boosting) | 24 fitur | `gradient_boosting_regressor_TUNED.joblib` (~330 KB) |
| LSTM | Deep Learning (RNN) | Window 24 jam (1 fitur) | `lstm_model.keras` (~400 KB) |

### Hasil Evaluasi (Full Test Set — 5.121 data points)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.0166 | 0.0249 | 0.9988 |
| **Gradient Boosting** | **0.0158** | **0.0239** | **0.9989** |
| LSTM | 0.3462 | 0.4896 | 0.5243 |

**Gradient Boosting** adalah model terbaik — unggul di semua metrik dan paling efisien (~330 KB).

## Fitur Website

- **Dashboard Smart Home** — Kartu statistik, tabel variabel dataset, grafik perbandingan 3 model, tabel metrik evaluasi
- **Prediksi & Perbandingan** — Filter rentang waktu, toggle model, grafik interaktif Plotly, metrik dinamis & resmi, analisis error, interpretasi hasil
- **Informasi Model** — Arsitektur, parameter, kelebihan/kekurangan, feature engineering, info dataset

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Framework | Streamlit (Python) |
| ML Models | scikit-learn, TensorFlow/Keras |
| Charts | Plotly |
| Data | Pandas, NumPy |

## Struktur Proyek

```
streamlit/
├── app.py                      # Aplikasi Streamlit utama
├── requirements.txt            # Dependencies Python
├── README.md
├── .gitignore
├── LICENSE
├── backend/
│   ├── models/                 # File model ML + dataset ZIP
│   │   ├── individual+household+electric+power+consumption.zip
│   │   ├── random_forest_regressor_TUNED.joblib
│   │   ├── gradient_boosting_regressor_TUNED.joblib
│   │   ├── lstm_model.keras
│   │   └── test_metrics_default.csv
│   ├── data/                   # Data hasil preprocessing
│   │   ├── sample_data.json
│   │   ├── full_test_data.json
│   │   └── scaler_params.json
│   ├── app/                    # Modul backend (legacy)
│   └── generate_sample_data.py
```

## Setup & Menjalankan

### Prasyarat

- Python 3.11+

### Instalasi

```bash
git clone <repository-url>
cd streamlit

python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Jalankan

```bash
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

### Deploy ke Streamlit Cloud

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repository, branch, dan file `app.py`
4. Deploy

## Catatan

- Semua data berasal dari dataset UCI asli, bukan data sintetis
- Folder proyek adalah `streamlit/`, bukan `Website/`
- Dataset ZIP (~20 MB) di-push ke GitHub. File TXT (~127 MB) di-gitignore
- Model terbaik: **Gradient Boosting** — MAE terendah, RMSE terendah, R² tertinggi, ukuran model terkecil

---

Skripsi — Franscen Yosafat Sinambela · 2025
