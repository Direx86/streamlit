import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for tensorflow (heavy)
_tf_loaded = False
_lstm_model = None


def _load_keras_model(path):
    global _tf_loaded
    if not _tf_loaded:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import load_model
    _tf_loaded = True
    return load_model(path)


class EnergyPredictor:
    """Loads and manages all 3 models for energy prediction."""

    def __init__(self, config):
        self.config = config
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        self.models_loaded = False

    def load_models(self):
        """Load all models into memory."""
        try:
            logger.info("Loading Random Forest model...")
            self.rf_model = joblib.load(self.config.RF_MODEL_PATH)
            logger.info("RF model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RF model: {e}")

        try:
            logger.info("Loading Gradient Boosting model...")
            logger.info(f"GB model path: {self.config.GB_MODEL_PATH}")
            logger.info(f"GB model file exists: {self.config.GB_MODEL_PATH.exists()}")
            if self.config.GB_MODEL_PATH.exists():
                logger.info(f"GB model file size: {self.config.GB_MODEL_PATH.stat().st_size} bytes")
            self.gb_model = joblib.load(self.config.GB_MODEL_PATH)
            logger.info("GB model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GB model: {e}", exc_info=True)

        try:
            logger.info("Loading LSTM model...")
            self.lstm_model = _load_keras_model(self.config.LSTM_MODEL_PATH)
            logger.info("LSTM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")

        self.models_loaded = True
        loaded = sum(1 for m in [self.rf_model, self.gb_model, self.lstm_model] if m is not None)
        logger.info(f"Models loaded: {loaded}/3")

    def _build_tree_features(self, readings: list[dict]) -> pd.DataFrame:
        """Build feature DataFrame for tree models from sensor readings."""
        df = pd.DataFrame(readings)

        target = "Global_active_power"
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"{target}_lag{lag}"] = df[target].shift(lag)

        # Rolling features (interleaved: rollmean then rollstd per window)
        for w in [3, 6, 12, 24]:
            df[f"{target}_rollmean{w}"] = df[target].rolling(w).mean()
            df[f"{target}_rollstd{w}"] = df[target].rolling(w).std()

        df = df.dropna()
        return df[self.config.TREE_FEATURE_COLS]

    def _scale_for_lstm(self, values: np.ndarray) -> np.ndarray:
        """Scale values using saved scaler parameters."""
        return (values - self.config.SCALER_MEAN) / self.config.SCALER_STD

    def _inverse_scale_lstm(self, values: np.ndarray) -> np.ndarray:
        """Inverse scale LSTM output."""
        return values * self.config.SCALER_STD + self.config.SCALER_MEAN

    def predict(self, readings: list[dict]) -> dict:
        """Run prediction using all 3 models."""
        results = {}

        # --- Tree models ---
        if self.rf_model or self.gb_model:
            tree_df = self._build_tree_features(readings)
            if len(tree_df) > 0:
                last_row = tree_df.iloc[[-1]]

                if self.rf_model:
                    rf_pred = self.rf_model.predict(last_row)
                    results["RandomForest"] = float(rf_pred[0])

                if self.gb_model:
                    gb_pred = self.gb_model.predict(last_row)
                    results["GradientBoosting"] = float(gb_pred[0])

        # --- LSTM model ---
        if self.lstm_model:
            window = self.config.LSTM_WINDOW
            gap_values = [r["Global_active_power"] for r in readings]

            if len(gap_values) >= window:
                last_window = np.array(gap_values[-window:])
                scaled = self._scale_for_lstm(last_window)
                X = scaled.reshape(1, window, 1)
                pred_scaled = self.lstm_model.predict(X, verbose=0)
                pred_value = self._inverse_scale_lstm(pred_scaled.ravel()[0])
                results["LSTM"] = float(pred_value)

        return results

    def predict_batch(self, readings: list[dict], n_predictions: int = 1) -> list[dict]:
        """Run predictions on a sliding window of readings."""
        all_predictions = []
        min_required = max(24, self.config.LSTM_WINDOW)

        for i in range(min_required, len(readings)):
            window_readings = readings[max(0, i - 48):i + 1]
            preds = self.predict(window_readings)
            if preds:
                entry = {"index": i}
                entry.update(preds)
                all_predictions.append(entry)

        return all_predictions

    def get_metrics(self) -> list[dict]:
        """Load metrics from CSV."""
        try:
            df = pd.read_csv(self.config.METRICS_PATH)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return []

    def get_sample_data(self) -> Optional[dict]:
        """Load pre-generated sample data (168 jam / 7 hari dari test set UCI asli)."""
        try:
            with open(self.config.SAMPLE_DATA_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return None

    def get_full_test_data(self) -> Optional[dict]:
        """Load full test set data (5121 data points dari dataset UCI asli)."""
        try:
            full_path = self.config.SAMPLE_DATA_PATH.parent / "full_test_data.json"
            with open(full_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load full test data: {e}")
            return None

    def get_models_info(self) -> list[dict]:
        """Return info about loaded models."""
        models_info = []

        models_info.append({
            "name": "Random Forest",
            "type": "Ensemble (Bagging)",
            "description": "Ensemble dari banyak Decision Tree yang dilatih secara paralel dengan teknik bagging. Setiap tree dilatih pada subset data acak, lalu hasil prediksi dirata-ratakan.",
            "parameters": {
                "n_estimators": "80-200 (tuned)",
                "max_depth": "8-12 atau None (tuned)",
                "random_state": 42,
            },
            "status": "loaded" if self.rf_model else "not loaded",
        })

        models_info.append({
            "name": "Gradient Boosting",
            "type": "Ensemble (Boosting)",
            "description": "Ensemble dari Decision Tree yang dilatih secara sekuensial. Setiap tree baru belajar dari kesalahan (residual) tree sebelumnya, sehingga model terus membaik.",
            "parameters": {
                "n_estimators": "120-360 (tuned)",
                "learning_rate": "0.05-0.20 (tuned)",
                "max_depth": 3,
                "random_state": 42,
            },
            "status": "loaded" if self.gb_model else "not loaded",
        })

        models_info.append({
            "name": "LSTM",
            "type": "Deep Learning (RNN)",
            "description": "Long Short-Term Memory, arsitektur Recurrent Neural Network yang mampu mempelajari pola temporal jangka panjang. Menggunakan window 24 jam untuk memprediksi 1 jam ke depan.",
            "parameters": {
                "layers": "LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)",
                "optimizer": "Adam (lr=0.001)",
                "window_size": 24,
                "loss": "MSE",
            },
            "status": "loaded" if self.lstm_model else "not loaded",
        })

        return models_info
