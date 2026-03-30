# Prediksi Konsumsi Energi Rumah Tangga Menggunakan Metode Random Forest, Gradient Boosting, dan LSTM pada Sistem Smart Home

Aplikasi web **Streamlit** untuk memprediksi dan membandingkan konsumsi energi rumah tangga menggunakan 3 metode machine learning, berbasis dataset UCI - Individual Household Electric Power Consumption.

> **Arsitektur:** Single repository Streamlit (menggabungkan frontend & backend dalam satu aplikasi Python). Tidak memerlukan Vercel atau Railway.

## Dataset

**Sumber:** [UCI Machine Learning Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

| Detail          | Keterangan                                                |
| --------------- | --------------------------------------------------------- |
| File            | `household_power_consumption.txt` (dari ZIP dalam repo)   |
| Jumlah Baris    | 2.075.259 (per menit)                                    |
| Periode         | Desember 2006 - November 2010                            |
| Resampling      | Rata-rata per jam (hourly) → 34.168 baris                |
| Split Data      | Train 70% (23.902) / Val 15% (5.121) / Test 15% (5.121)  |

Dataset asli berukuran ~127 MB (melebihi batas GitHub 100 MB), sehingga disimpan dalam bentuk ZIP (~20 MB) di `backend/models/individual+household+electric+power+consumption.zip`. Aplikasi Streamlit akan mengekstrak file ini secara otomatis saat pertama kali dijalankan.

### Variabel Dataset

| Variabel               | Unit    | Deskripsi                                            |
| ---------------------- | ------- | ---------------------------------------------------- |
| Global_active_power    | kW      | Total daya aktif yang dikonsumsi rumah tangga (**TARGET**) |
| Global_reactive_power  | kW      | Total daya reaktif yang dikonsumsi rumah tangga      |
| Voltage                | Volt    | Tegangan listrik rata-rata per menit                 |
| Global_intensity       | Ampere  | Arus listrik rata-rata per menit                     |
| Sub_metering_1         | Wh      | Dapur: dishwasher, oven, microwave                   |
| Sub_metering_2         | Wh      | Laundry: mesin cuci, pengering, lampu, kulkas        |
| Sub_metering_3         | Wh      | Pemanas air listrik & AC                             |

## Model Machine Learning

### 1. Random Forest (TUNED)
- **Tipe:** Ensemble (Bagging)
- **Input:** 24 fitur (6 sensor + 4 kalender + 6 lag + 8 rolling)
- **Tuning:** RandomizedSearchCV (n_estimators: 80–200, max_depth: 8/12/None)
- **File:** `random_forest_regressor_TUNED.joblib` (~48 MB)

### 2. Gradient Boosting (TUNED)
- **Tipe:** Ensemble (Boosting)
- **Input:** 24 fitur (6 sensor + 4 kalender + 6 lag + 8 rolling)
- **Tuning:** RandomizedSearchCV (n_estimators: 120–360, learning_rate: 0.05–0.20)
- **File:** `gradient_boosting_regressor_TUNED.joblib` (~330 KB)

### 3. LSTM
- **Tipe:** Deep Learning (Recurrent Neural Network)
- **Input:** Window 24 jam Global_active_power (1 fitur, StandardScaler normalized)
- **Arsitektur:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16, ReLU) → Dense(1)
- **Training:** Adam (lr=0.001), MSE loss, Early Stopping (patience=10), 50 epochs
- **File:** `lstm_model.keras` (~400 KB)

### Feature Engineering (Random Forest & Gradient Boosting — 24 fitur)

| Kategori         | Jumlah | Fitur                                                                     |
| ---------------- | ------ | ------------------------------------------------------------------------- |
| Sensor           | 6      | Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3     |
| Kalender         | 4      | hour, dayofweek, is_weekend, month                                       |
| Lag              | 6      | lag1, lag2, lag3, lag6, lag12, lag24                                      |
| Rolling          | 8      | rollmean3/6/12/24, rollstd3/6/12/24                                     |

### Input LSTM

```
Window 24 jam Global_active_power (StandardScaler normalized) — Shape: (1, 24, 1)
Scaler: mean = 1.086397, std = 0.929282 (dihitung dari training set)
```

## Metrik Evaluasi

### Penjelasan Metrik

| Metrik | Nama Lengkap                      | Rumus                                     | Interpretasi                                                                                     |
| ------ | --------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **MAE**  | Mean Absolute Error               | (1/n) × Σ\|yᵢ − ŷᵢ\|                      | Rata-rata kesalahan absolut dalam satuan kW. **Semakin kecil = semakin akurat.**                  |
| **RMSE** | Root Mean Squared Error           | √[(1/n) × Σ(yᵢ − ŷᵢ)²]                   | Akar rata-rata kesalahan kuadrat (kW). Lebih sensitif terhadap kesalahan besar. **Semakin kecil = semakin konsisten.** |
| **R²**   | R-Squared (Koefisien Determinasi) | 1 − (SS_res / SS_tot)                     | Proporsi variasi data yang dijelaskan oleh model (0–1). **Semakin mendekati 1 = semakin baik.**  |

### Hasil Evaluasi — Full Test Set (5.121 data points)

| Model               | MAE      | RMSE     | R²       | Keterangan                           |
| -------------------- | -------- | -------- | -------- | ------------------------------------ |
| **Random Forest**    | 0.0166   | 0.0249   | 0.9988   | Performa sangat baik                 |
| **Gradient Boosting**| 0.0158   | 0.0239   | 0.9989   | **Terbaik di semua metrik**          |
| **LSTM**             | 0.3462   | 0.4896   | 0.5243   | Performa lebih rendah (1 fitur saja) |

### Interpretasi Hasil

**Random Forest & Gradient Boosting** menunjukkan performa yang sangat baik:
- R² mendekati 1.0 (> 0.998), artinya model mampu menjelaskan lebih dari 99.8% variasi konsumsi energi
- MAE sangat kecil (~0.016 kW), artinya rata-rata prediksi meleset hanya ~16 Watt dari nilai sebenarnya
- RMSE mendekati MAE, menunjukkan prediksi **konsisten tanpa kesalahan ekstrem**
- Keunggulan ini berkat **24 fitur** yang mencakup sensor, kalender, lag, dan rolling statistics

**Gradient Boosting menjadi model terbaik** karena:
1. **MAE terendah** (0.0158 vs 0.0166) — rata-rata error ~5% lebih kecil dari RF
2. **RMSE terendah** (0.0239 vs 0.0249) — lebih konsisten, kesalahan besar lebih jarang
3. **R² tertinggi** (0.9989 vs 0.9988) — menjelaskan variasi data sedikit lebih baik
4. **Ukuran model lebih kecil** (~330 KB vs ~48 MB) — lebih efisien untuk deployment
5. Metode boosting (sequential learning) secara iteratif mengoreksi kesalahan tree sebelumnya, menghasilkan akurasi lebih tinggi dibanding bagging (RF)

**LSTM memiliki performa lebih rendah (R² = 0.5243)** karena:
- Hanya menggunakan **1 fitur** (Global_active_power), tanpa sensor lain, informasi waktu, atau fitur lag/rolling
- Namun tetap mampu menangkap **tren umum** naik-turunnya konsumsi energi harian
- Berfungsi sebagai **baseline univariate** untuk menunjukkan pentingnya feature engineering

**Kesimpulan:** Gradient Boosting adalah model yang paling optimal untuk prediksi konsumsi energi rumah tangga dalam penelitian ini, berdasarkan keunggulan di seluruh metrik evaluasi (MAE, RMSE, R²) serta efisiensi ukuran model.

## Struktur Proyek

```
Website/
├── app.py                          # Aplikasi Streamlit (frontend + backend dalam 1 file)
├── requirements.txt                # Dependencies Python
├── .gitignore
├── README.md
├── backend/
│   ├── models/                     # File model ML + dataset ZIP
│   │   ├── individual+household+electric+power+consumption.zip  # Dataset UCI (~20 MB)
│   │   ├── random_forest_regressor_TUNED.joblib                 # Model RF (~48 MB)
│   │   ├── gradient_boosting_regressor_TUNED.joblib             # Model GB (~330 KB)
│   │   ├── lstm_model.keras                                     # Model LSTM (~400 KB)
│   │   └── test_metrics_default.csv                             # Metrik evaluasi
│   ├── data/                       # Data hasil preprocessing dari dataset UCI asli
│   │   ├── sample_data.json            # 168 data points (7 hari terakhir test set)
│   │   ├── full_test_data.json         # 5.121 data points (seluruh test set)
│   │   └── scaler_params.json          # Parameter StandardScaler LSTM
│   ├── app/                        # Modul backend (FastAPI — opsional/legacy)
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── predictor.py
│   │   └── schemas.py
│   └── generate_sample_data.py     # Script proses dataset UCI → JSON
└── household_power_consumption.txt # Dataset UCI hasil ekstrak (otomatis, di-gitignore)
```

### Ukuran File & Git

| File                                       | Ukuran  | Di-push ke GitHub? |
| ------------------------------------------ | ------- | ------------------ |
| `individual+household+electric+power+consumption.zip` | ~20 MB  | **Ya** (ZIP)       |
| `random_forest_regressor_TUNED.joblib`     | ~48 MB  | Ya                 |
| `gradient_boosting_regressor_TUNED.joblib` | ~330 KB | Ya                 |
| `lstm_model.keras`                         | ~400 KB | Ya                 |
| `data/sample_data.json`                    | ~27 KB  | Ya                 |
| `data/full_test_data.json`                 | ~569 KB | Ya                 |
| `household_power_consumption.txt`          | ~127 MB | **Tidak** (`.gitignore`) |

Dataset ZIP (~20 MB) sudah di bawah batas GitHub 100 MB, sehingga bisa di-push. File TXT hasil ekstrak (~127 MB) otomatis di-gitignore.

## Tech Stack

| Komponen    | Teknologi                                |
| ----------- | ---------------------------------------- |
| Framework   | **Streamlit** (Python)                   |
| ML Models   | scikit-learn, TensorFlow/Keras           |
| Charts      | Plotly                                   |
| Data        | Pandas, NumPy                            |
| Dataset     | UCI Machine Learning Repository          |

## Fitur Website

### 1. Dashboard Smart Home
- 4 kartu statistik: konsumsi terakhir, rata-rata, tertinggi, terendah
- Tabel 7 variabel dataset UCI beserta deskripsi dan satuan (dengan link ke sumber)
- Grafik perbandingan prediksi 3 model vs data aktual (168 jam)
- Tabel metrik evaluasi dengan penanda "Best" per metrik

### 2. Prediksi & Perbandingan
- Filter rentang waktu: 24 Jam, 48 Jam, 7 Hari, 30 Hari, Full Test Set
- Toggle show/hide per model (Aktual, RF, GB, LSTM)
- Grafik interaktif Plotly dengan tooltip
- **Dua tabel metrik:**
  - Metrik Dinamis: dihitung ulang berdasarkan rentang waktu yang dipilih
  - Metrik Resmi (Full Test Set 5.121 points): angka tetap untuk dilaporkan di skripsi
- Analisis error per model (MAE, RMSE, Max Error)
- **Penjelasan detail metrik evaluasi** (apa itu MAE, RMSE, R² dan cara membacanya)
- **Interpretasi hasil lengkap** dengan perbandingan antar model dan kesimpulan

### 3. Informasi Model
- Detail arsitektur tiap model (RF, GB, LSTM)
- Parameter dan hyperparameter
- Kelebihan dan kekurangan masing-masing model
- Rekayasa fitur (feature engineering) — 24 fitur untuk tree models
- Informasi dataset dengan link ke UCI Repository

## Setup & Menjalankan Aplikasi

### Prasyarat

- Python 3.11+
- File model ML (dari Google Colab notebook):
  - `random_forest_regressor_TUNED.joblib`
  - `gradient_boosting_regressor_TUNED.joblib`
  - `lstm_model.keras`
  - `test_metrics_default.csv`

### Instalasi

```bash
# Clone repository
git clone <repository-url>
cd Website

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Data (Opsional — hanya jika file JSON belum ada)

Jika folder `backend/data/` belum berisi `sample_data.json` dan `full_test_data.json`:

```bash
cd backend

# Pastikan dataset zip ada di backend/models/
# Ekstrak manual atau biarkan app.py mengekstrak otomatis
# Jika ingin generate ulang:
python generate_sample_data.py

cd ..
```

### Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Deploy ke Streamlit Cloud (Opsional)

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repository, branch, dan file `app.py`
4. Deploy — Streamlit Cloud akan otomatis install `requirements.txt`

> **Catatan:** Pastikan file model (`*.joblib`, `*.keras`) dan dataset ZIP sudah ada di repository. Total ukuran repo ~70 MB, masih dalam batas GitHub.

## Catatan untuk Sidang Skripsi

- **Dataset:** Seluruh data berasal dari [dataset UCI asli](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), bukan data sintetis
- **Satuan kW:** Kolom `Global_active_power` di dataset UCI memang dalam kilowatt
- **Grafik:** 4 garis — Data Aktual (biru), Random Forest (hijau), Gradient Boosting (oranye), LSTM (merah)
- **Tabel metrik "Best":** Menandai model terbaik per metrik (MAE, RMSE, R²)
- **Dua tabel metrik di halaman Prediksi:**
  - Dinamis: berubah sesuai rentang waktu yang dipilih
  - Resmi: tetap dari full test set (5.121 points) — **ini yang dilaporkan di skripsi**
- **LSTM R² rendah (~0.52):** Hanya menggunakan 1 fitur (Global_active_power), sedangkan RF dan GB menggunakan 24 fitur
- **Model terbaik:** Gradient Boosting — unggul di semua metrik (MAE, RMSE, R²) dan efisien (330 KB)
- **Framework:** Streamlit (Python) — single repository, tidak memerlukan deploy terpisah untuk frontend dan backend
- **Dataset ZIP:** File dataset disimpan dalam bentuk ZIP (~20 MB) agar bisa di-push ke GitHub. Aplikasi mengekstrak otomatis saat dijalankan
