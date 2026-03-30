from pydantic import BaseModel
from typing import List, Optional


class SensorInput(BaseModel):
    """Single timestep sensor reading from Smart Home."""
    Global_active_power: float
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float
    hour: int
    dayofweek: int
    month: int


class PredictionRequest(BaseModel):
    """Request body: 24+ hourly readings for prediction."""
    readings: List[SensorInput]


class SinglePrediction(BaseModel):
    model_name: str
    predicted_value: float


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    input_hours: int


class MetricRow(BaseModel):
    model: str
    MAE: float
    RMSE: float
    R2: float


class MetricsResponse(BaseModel):
    metrics: List[MetricRow]


class SampleDataPoint(BaseModel):
    datetime: str
    actual: float
    rf_pred: float
    gb_pred: float
    lstm_pred: float


class SampleDataResponse(BaseModel):
    data: List[SampleDataPoint]
    metrics: List[MetricRow]


class ModelInfo(BaseModel):
    name: str
    type: str
    description: str
    parameters: dict
    status: str


class ModelsInfoResponse(BaseModel):
    models: List[ModelInfo]
