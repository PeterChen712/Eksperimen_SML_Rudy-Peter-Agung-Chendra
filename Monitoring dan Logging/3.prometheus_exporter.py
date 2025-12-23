from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
from prometheus_client.core import CollectorRegistry
import time
import psutil
import os

PREDICTION_COUNTER = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Time spent processing prediction request',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_LATENCY_SUMMARY = Summary(
    'ml_prediction_latency_summary_seconds',
    'Summary of prediction latency',
    ['model_name']
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score',
    ['model_name', 'version']
)

MODEL_F1_SCORE = Gauge(
    'ml_model_f1_score',
    'Current model F1 score',
    ['model_name', 'version']
)

PREDICTION_CLASS = Counter(
    'ml_prediction_class_total',
    'Distribution of predicted classes',
    ['model_name', 'predicted_class']
)

INPUT_FEATURE_VALUE = Histogram(
    'ml_input_feature_value',
    'Distribution of input feature values',
    ['feature_name'],
    buckets=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
)

MODEL_LOAD_TIME = Gauge(
    'ml_model_load_time_seconds',
    'Time taken to load the model',
    ['model_name']
)

ACTIVE_REQUESTS = Gauge(
    'ml_active_requests',
    'Number of requests currently being processed'
)

ERROR_COUNTER = Counter(
    'ml_errors_total',
    'Total number of errors',
    ['model_name', 'error_type']
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Current memory usage in bytes',
    ['type']
)

CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

MODEL_INFO = Info(
    'ml_model',
    'Information about the ML model'
)

DATA_DRIFT_SCORE = Gauge(
    'ml_data_drift_score',
    'Data drift score',
    ['feature_name']
)

PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Distribution of prediction confidence scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


class MetricsCollector:
    def __init__(self, model_name="heart-disease-classifier", model_version="1.0"):
        self.model_name = model_name
        self.model_version = model_version
        MODEL_INFO.info({
            'name': model_name,
            'version': model_version,
            'author': 'Rudy Peter Agung Chendra',
            'dicoding_id': 'M208D5Y1771',
            'framework': 'scikit-learn',
            'type': 'RandomForestClassifier'
        })
        
    def record_prediction(self, start_time, predicted_class, confidence, success=True):
        latency = time.time() - start_time
        status = 'success' if success else 'error'
        PREDICTION_COUNTER.labels(model_name=self.model_name, status=status).inc()
        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(latency)
        PREDICTION_LATENCY_SUMMARY.labels(model_name=self.model_name).observe(latency)
        PREDICTION_CLASS.labels(model_name=self.model_name, predicted_class=str(predicted_class)).inc()
        PREDICTION_CONFIDENCE.labels(model_name=self.model_name).observe(confidence)
        
    def record_input_features(self, features):
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        for i, value in enumerate(features):
            if i < len(feature_names):
                INPUT_FEATURE_VALUE.labels(feature_name=feature_names[i]).observe(value)
                
    def record_error(self, error_type):
        ERROR_COUNTER.labels(model_name=self.model_name, error_type=error_type).inc()
        
    def update_model_metrics(self, accuracy, f1_score):
        MODEL_ACCURACY.labels(model_name=self.model_name, version=self.model_version).set(accuracy)
        MODEL_F1_SCORE.labels(model_name=self.model_name, version=self.model_version).set(f1_score)
        
    def update_system_metrics(self):
        memory = psutil.virtual_memory()
        MEMORY_USAGE.labels(type='used').set(memory.used)
        MEMORY_USAGE.labels(type='available').set(memory.available)
        MEMORY_USAGE.labels(type='total').set(memory.total)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        CPU_USAGE.set(cpu_percent)
        
    def set_model_load_time(self, load_time):
        MODEL_LOAD_TIME.labels(model_name=self.model_name).set(load_time)
        
    def update_drift_score(self, feature_name, score):
        DATA_DRIFT_SCORE.labels(feature_name=feature_name).set(score)
        
    def increment_active_requests(self):
        ACTIVE_REQUESTS.inc()
        
    def decrement_active_requests(self):
        ACTIVE_REQUESTS.dec()


def get_metrics():
    return generate_latest(REGISTRY)


if __name__ == "__main__":
    collector = MetricsCollector()
    collector.update_model_metrics(accuracy=0.96, f1_score=0.95)
    collector.set_model_load_time(1.5)
    collector.update_system_metrics()
    import random
    for _ in range(10):
        start = time.time()
        time.sleep(random.uniform(0.01, 0.1))
        collector.record_prediction(
            start_time=start,
            predicted_class=random.choice([0, 1, 2]),
            confidence=random.uniform(0.7, 1.0)
        )
        collector.record_input_features([
            random.uniform(4.0, 8.0),
            random.uniform(2.0, 4.5),
            random.uniform(1.0, 7.0),
            random.uniform(0.1, 2.5)
        ])
    print("Prometheus Metrics:")
    print(get_metrics().decode('utf-8'))
