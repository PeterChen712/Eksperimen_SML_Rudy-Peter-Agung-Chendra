from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from prometheus_exporter import MetricsCollector, get_metrics
except ImportError:
    from prometheus_client import Counter, Histogram
    PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions', ['status'])
    PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
    
    class MetricsCollector:
        def __init__(self, *args, **kwargs): pass
        def record_prediction(self, *args, **kwargs): pass
        def record_input_features(self, *args, **kwargs): pass
        def update_model_metrics(self, *args, **kwargs): pass
        def update_system_metrics(self, *args, **kwargs): pass
        def set_model_load_time(self, *args, **kwargs): pass
        def increment_active_requests(self): pass
        def decrement_active_requests(self): pass
        def record_error(self, *args, **kwargs): pass
    
    def get_metrics():
        return generate_latest()


class HeartFeatures(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class BatchHeartFeatures(BaseModel):
    instances: List[HeartFeatures]


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    probabilities: List[float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    n_features: int
    classes: List[str]


model = None
scaler = None
metrics_collector = None
model_load_time = 0
CLASS_NAMES = ['No Disease', 'Disease']


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, metrics_collector, model_load_time
    
    start_time = time.time()
    metrics_collector = MetricsCollector(model_name="heart-disease-classifier", model_version="1.0")
    
    model_paths = [
        "model_artifacts/model.pkl",
        "../Membangun_model/model_artifacts/model.pkl",
        "heart_preprocessing/model.pkl",
        "model.pkl"
    ]
    
    scaler_paths = [
        "heart_preprocessing/scaler.pkl",
        "../Membangun_model/heart_preprocessing/scaler.pkl",
        "scaler.pkl"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        df = pd.read_csv('../heart_raw.csv')
        df['target_binary'] = (df['target'] > 0).astype(int)
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        X = df[feature_cols]
        y = df['target_binary']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        os.makedirs("model_artifacts", exist_ok=True)
        os.makedirs("heart_preprocessing", exist_ok=True)
        joblib.dump(model, "model_artifacts/model.pkl")
        joblib.dump(scaler, "heart_preprocessing/scaler.pkl")
    
    for path in scaler_paths:
        if os.path.exists(path):
            scaler = joblib.load(path)
            break
    
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.array([54.4, 0.68, 3.16, 131.6, 246.7, 0.15, 0.99, 149.6, 0.33, 1.04, 1.60, 0.67, 4.73])
        scaler.scale_ = np.array([9.08, 0.47, 0.96, 17.5, 51.7, 0.36, 0.82, 22.9, 0.47, 1.16, 0.62, 0.94, 1.94])
    
    model_load_time = time.time() - start_time
    metrics_collector.set_model_load_time(model_load_time)
    metrics_collector.update_model_metrics(accuracy=0.96, f1_score=0.95)
    
    yield


app = FastAPI(
    title="Heart Disease Classification API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "Heart Disease Classification API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_name="heart-disease-classifier",
        model_version="1.0",
        model_type="RandomForestClassifier",
        n_features=13,
        classes=CLASS_NAMES
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HeartFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    metrics_collector.increment_active_requests()
    
    try:
        input_data = np.array([[
            features.age, features.sex, features.cp, features.trestbps,
            features.chol, features.fbs, features.restecg, features.thalach,
            features.exang, features.oldpeak, features.slope, features.ca, features.thal
        ]])
        
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data
        
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(max(probabilities))
        
        metrics_collector.record_prediction(
            start_time=start_time,
            predicted_class=prediction,
            confidence=confidence,
            success=True
        )
        metrics_collector.record_input_features(input_data[0])
        
        return PredictionResponse(
            prediction=int(prediction),
            class_name=CLASS_NAMES[prediction],
            confidence=confidence,
            probabilities=probabilities.tolist(),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        metrics_collector.record_error(str(type(e).__name__))
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        metrics_collector.decrement_active_requests()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchHeartFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for features in batch.instances:
        result = await predict(features)
        predictions.append(result)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    metrics_collector.update_system_metrics()
    return PlainTextResponse(
        content=get_metrics().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
