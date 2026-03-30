import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.predictor import EnergyPredictor
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    SinglePrediction,
    MetricsResponse,
    MetricRow,
    SampleDataResponse,
    SampleDataPoint,
    ModelsInfoResponse,
    ModelInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = EnergyPredictor(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Starting up - loading models...")
    predictor.load_models()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Home Energy Prediction API",
    description=(
        "API untuk Prediksi Konsumsi Energi Rumah Tangga menggunakan "
        "Random Forest, Gradient Boosting, dan LSTM"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.FRONTEND_URL,
        "https://prediksi-konsumsi-energi-rumah-tangga.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health ---
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": predictor.models_loaded,
        "rf": predictor.rf_model is not None,
        "gb": predictor.gb_model is not None,
        "lstm": predictor.lstm_model is not None,
    }


# --- Metrics ---
@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    rows = predictor.get_metrics()
    if not rows:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return MetricsResponse(metrics=[MetricRow(**r) for r in rows])


# --- Model Info ---
@app.get("/api/models/info", response_model=ModelsInfoResponse)
def get_models_info():
    infos = predictor.get_models_info()
    return ModelsInfoResponse(models=[ModelInfo(**i) for i in infos])


# --- Predict ---
@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if len(req.readings) < 25:
        raise HTTPException(
            status_code=400,
            detail=f"Minimal 25 data reading diperlukan (diterima: {len(req.readings)}). "
            "Butuh 24 data historis + 1 data terbaru untuk fitur lag/rolling.",
        )

    readings_dicts = [r.model_dump() for r in req.readings]
    results = predictor.predict(readings_dicts)

    if not results:
        raise HTTPException(status_code=500, detail="Prediction failed for all models")

    preds = [
        SinglePrediction(model_name=name, predicted_value=round(val, 6))
        for name, val in results.items()
    ]
    return PredictionResponse(predictions=preds, input_hours=len(req.readings))


# --- Sample Data (Demo - 168 jam / 7 hari) ---
@app.get("/api/sample-data", response_model=SampleDataResponse)
def get_sample_data():
    data = predictor.get_sample_data()
    if not data:
        raise HTTPException(status_code=404, detail="Sample data not found")

    return SampleDataResponse(
        data=[SampleDataPoint(**d) for d in data["data"]],
        metrics=[MetricRow(**m) for m in data["metrics"]],
    )


# --- Full Test Data (5121 data points) ---
@app.get("/api/full-test-data")
def get_full_test_data(
    last_n: Optional[int] = Query(None, description="Ambil N data terakhir saja"),
):
    data = predictor.get_full_test_data()
    if not data:
        raise HTTPException(status_code=404, detail="Full test data not found")

    points = data["data"]
    if last_n and last_n > 0:
        points = points[-last_n:]

    return {
        "data": points,
        "metrics": data["metrics"],
        "total_points": len(data["data"]),
    }


# --- Dataset Info ---
@app.get("/api/dataset-info")
def get_dataset_info():
    return {
        "name": "UCI Individual Household Electric Power Consumption",
        "source": "UCI Machine Learning Repository",
        "period": "Desember 2006 - November 2010",
        "original_resolution": "Per menit",
        "resampled_resolution": "Per jam (1H) - rata-rata",
        "total_hourly_records": 34168,
        "split": {
            "train": {"records": 23902, "percentage": "70%"},
            "validation": {"records": 5121, "percentage": "15%"},
            "test": {"records": 5121, "percentage": "15%"},
        },
        "target_variable": "Global_active_power (kW)",
        "features_tree_models": 24,
        "lstm_window": 24,
    }
