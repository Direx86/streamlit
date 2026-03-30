"""
Script untuk generate sample data dari dataset UCI ASLI + prediksi model ASLI.
Jalankan script ini SETELAH model files dan household_power_consumption.txt
ada di folder backend/models/

Usage:
    cd backend
    python generate_sample_data.py

Script ini akan:
1. Load dataset UCI household_power_consumption.txt yang asli
2. Preprocessing: resample 1 jam, feature engineering (sama persis dengan notebook)
3. Split data: Train 70% | Val 15% | Test 15% (time-based)
4. Jalankan prediksi ketiga model pada TEST SET
5. Hitung metrik evaluasi ASLI (MAE, RMSE, R²)
6. Simpan ke data/sample_data.json untuk ditampilkan di website
7. Simpan scaler parameters ke data/scaler_params.json
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent))
from app import config

# ─── CONFIG (sama persis dengan notebook) ─────────────────
TARGET_COL = "Global_active_power"
RESAMPLE_RULE = "1h"
TEST_SIZE_RATIO = 0.15
VAL_SIZE_RATIO = 0.15
LSTM_WINDOW = 24
DISPLAY_HOURS = 168  # 7 hari terakhir dari test set untuk ditampilkan di website


# ─── FUNGSI PREPROCESSING (sama persis dengan notebook) ───
def load_uci_txt(txt_path: str) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep=";", na_values=["?", "NA", "NaN", ""], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce", dayfirst=True)
    df = df.assign(datetime=dt).dropna(subset=["datetime"]).drop(columns=["Date", "Time"])
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample(RESAMPLE_RULE).mean(numeric_only=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = idx.month
    return out


def add_lag_rolling_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    # Interleaved: rollmean then rollstd per window (matches notebook training order)
    for w in [3, 6, 12, 24]:
        out[f"{target}_rollmean{w}"] = out[target].rolling(w).mean()
        out[f"{target}_rollstd{w}"] = out[target].rolling(w).std()
    return out.dropna()


def time_based_split(df, test_ratio=0.15, val_ratio=0.15):
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def make_supervised_sequence(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ─── MAIN ─────────────────────────────────────────────────
def main():
    dataset_path = config.MODELS_DIR / "household_power_consumption.txt"
    if not dataset_path.exists():
        print(f"ERROR: Dataset tidak ditemukan di {dataset_path}")
        print("Letakkan file household_power_consumption.txt di folder backend/models/")
        sys.exit(1)

    # ── 1. Load & Preprocess Dataset ──
    print("=" * 60)
    print("STEP 1: Loading dataset UCI asli...")
    df_raw = load_uci_txt(str(dataset_path))
    print(f"  Raw data: {len(df_raw)} baris ({df_raw.index.min()} - {df_raw.index.max()})")

    print("  Resampling ke 1 jam (mean)...")
    df_hour = resample_hourly(df_raw).dropna(subset=[TARGET_COL])
    print(f"  Data per jam: {len(df_hour)} baris")

    # ── 2. Feature Engineering untuk Tree Models ──
    print("\nSTEP 2: Feature engineering untuk RF & GB...")
    df_feat = add_calendar_features(df_hour)
    df_tree = add_lag_rolling_features(df_feat, TARGET_COL)
    print(f"  Data setelah feature engineering: {len(df_tree)} baris, {len(df_tree.columns)} kolom")

    # ── 3. Split Data ──
    print("\nSTEP 3: Time-based split...")
    train_tree, val_tree, test_tree = time_based_split(df_tree, TEST_SIZE_RATIO, VAL_SIZE_RATIO)
    print(f"  Train: {len(train_tree)} | Val: {len(val_tree)} | Test: {len(test_tree)}")
    print(f"  Test period: {test_tree.index.min()} - {test_tree.index.max()}")

    X_test_tree = test_tree.drop(columns=[TARGET_COL])
    y_test_tree = test_tree[TARGET_COL]

    # ── 4. Persiapan Data LSTM (sama persis notebook) ──
    print("\nSTEP 4: Persiapan data LSTM...")
    series = df_hour[[TARGET_COL]].dropna().copy()
    scaler_y = StandardScaler()

    # Split LSTM dulu sebelum fit scaler (fit hanya pada train)
    train_lstm, val_lstm, test_lstm = time_based_split(series, TEST_SIZE_RATIO, VAL_SIZE_RATIO)

    # Fit scaler HANYA pada training data (best practice)
    scaler_y.fit(train_lstm[[TARGET_COL]])
    scaler_mean = float(scaler_y.mean_[0])
    scaler_std = float(scaler_y.scale_[0])
    print(f"  Scaler: mean={scaler_mean:.6f}, std={scaler_std:.6f}")

    # Scale semua data
    series[TARGET_COL] = scaler_y.transform(series[[TARGET_COL]])
    train_lstm_s, val_lstm_s, test_lstm_s = time_based_split(series, TEST_SIZE_RATIO, VAL_SIZE_RATIO)

    # Buat sequence LSTM untuk test set (sama persis notebook)
    combined_for_test = pd.concat([train_lstm_s.tail(LSTM_WINDOW), val_lstm_s, test_lstm_s])
    X_test_l, y_test_l = make_supervised_sequence(combined_for_test[TARGET_COL].values, LSTM_WINDOW)

    # Ambil hanya bagian test
    n_val_points = len(val_lstm_s)
    X_test_lstm_only = X_test_l[n_val_points:]
    y_test_lstm_only = y_test_l[n_val_points:]

    print(f"  LSTM test sequences: {len(X_test_lstm_only)}")

    # ── 5. Load Models & Predict ──
    print("\nSTEP 5: Loading models & menjalankan prediksi...")

    # Random Forest
    print("  Loading Random Forest...")
    rf_model = joblib.load(config.RF_MODEL_PATH)
    rf_test_pred = rf_model.predict(X_test_tree[config.TREE_FEATURE_COLS])
    print(f"  RF predictions: {len(rf_test_pred)}")

    # Gradient Boosting
    print("  Loading Gradient Boosting...")
    gb_model = joblib.load(config.GB_MODEL_PATH)
    gb_test_pred = gb_model.predict(X_test_tree[config.TREE_FEATURE_COLS])
    print(f"  GB predictions: {len(gb_test_pred)}")

    # LSTM
    print("  Loading LSTM...")
    from tensorflow.keras.models import load_model
    lstm_model = load_model(config.LSTM_MODEL_PATH)

    lstm_test_pred_scaled = lstm_model.predict(
        X_test_lstm_only.reshape(-1, LSTM_WINDOW, 1), verbose=0
    ).ravel()
    lstm_test_pred = scaler_y.inverse_transform(
        lstm_test_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_test_lstm_true = scaler_y.inverse_transform(
        y_test_lstm_only.reshape(-1, 1)
    ).ravel()
    print(f"  LSTM predictions: {len(lstm_test_pred)}")

    # ── 6. Hitung Metrik Evaluasi ASLI ──
    print("\nSTEP 6: Menghitung metrik evaluasi...")
    rf_mae, rf_rmse, rf_r2 = regression_report(y_test_tree.values, rf_test_pred)
    gb_mae, gb_rmse, gb_r2 = regression_report(y_test_tree.values, gb_test_pred)
    lstm_mae, lstm_rmse, lstm_r2 = regression_report(y_test_lstm_true, lstm_test_pred)

    metrics = [
        {"model": "RandomForest", "MAE": round(rf_mae, 6), "RMSE": round(rf_rmse, 6), "R2": round(rf_r2, 6)},
        {"model": "GradientBoosting", "MAE": round(gb_mae, 6), "RMSE": round(gb_rmse, 6), "R2": round(gb_r2, 6)},
        {"model": "LSTM", "MAE": round(lstm_mae, 6), "RMSE": round(lstm_rmse, 6), "R2": round(lstm_r2, 6)},
    ]

    print()
    print("  Model               | MAE      | RMSE     | R2")
    print("  --------------------|----------|----------|----------")
    for m in metrics:
        print(f"  {m['model']:<19} | {m['MAE']:<8.4f} | {m['RMSE']:<8.4f} | {m['R2']:<8.4f}")

    # ── 7. Buat sample_data.json dari DATA TEST SET ASLI ──
    print(f"\nSTEP 7: Menyusun data untuk website ({DISPLAY_HOURS} jam terakhir test set)...")

    # Untuk RF & GB: ambil N jam terakhir test set
    n_display = min(DISPLAY_HOURS, len(test_tree))
    tree_display = test_tree.iloc[-n_display:]
    rf_display = rf_test_pred[-n_display:]
    gb_display = gb_test_pred[-n_display:]

    # Untuk LSTM: align berdasarkan datetime index
    lstm_pred_series = pd.Series(
        lstm_test_pred[-len(test_lstm):],
        index=test_lstm.index[-len(lstm_test_pred):]
    )

    sample_data = []
    for i, (idx, row) in enumerate(tree_display.iterrows()):
        actual = float(row[TARGET_COL])
        rf_val = float(rf_display[i])
        gb_val = float(gb_display[i])

        # LSTM: cari berdasarkan datetime index
        lstm_val = float(lstm_pred_series.get(idx, 0.0))

        sample_data.append({
            "datetime": idx.isoformat(),
            "actual": round(actual, 4),
            "rf_pred": round(rf_val, 4),
            "gb_pred": round(gb_val, 4),
            "lstm_pred": round(lstm_val, 4),
        })

    # ── 8. Simpan ke JSON ──
    output = {"data": sample_data, "metrics": metrics}
    out_path = config.SAMPLE_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({len(sample_data)} data points)")

    # ── 9. Simpan Scaler Parameters ──
    scaler_path = config.SAMPLE_DATA_PATH.parent / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump({"mean": scaler_mean, "std": scaler_std}, f, indent=2)
    print(f"  Saved: {scaler_path}")
    print(f"\n  PENTING: Update SCALER_MEAN={scaler_mean:.6f} dan SCALER_STD={scaler_std:.6f}")
    print(f"  di file app/config.py atau set sebagai environment variable.")

    # ── 10. Simpan juga full test set data untuk API predict ──
    full_test_data = []
    for i, (idx, row) in enumerate(test_tree.iterrows()):
        actual = float(row[TARGET_COL])
        rf_val = float(rf_test_pred[i])
        gb_val = float(gb_test_pred[i])
        lstm_val = float(lstm_pred_series.get(idx, 0.0))

        full_test_data.append({
            "datetime": idx.isoformat(),
            "actual": round(actual, 4),
            "rf_pred": round(rf_val, 4),
            "gb_pred": round(gb_val, 4),
            "lstm_pred": round(lstm_val, 4),
        })

    full_path = config.SAMPLE_DATA_PATH.parent / "full_test_data.json"
    with open(full_path, "w") as f:
        json.dump({"data": full_test_data, "metrics": metrics}, f)
    print(f"  Saved: {full_path} ({len(full_test_data)} data points - full test set)")

    print("\n" + "=" * 60)
    print("SELESAI! Data website sudah diperbarui dengan dataset UCI asli.")
    print("=" * 60)


if __name__ == "__main__":
    main()
