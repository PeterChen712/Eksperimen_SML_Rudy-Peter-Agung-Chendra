from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info
import time
import random
import psutil
import os
from datetime import datetime

PREDICTIONS_TOTAL = Counter(
    'predictions_total', 
    'Total number of predictions made',
    ['model', 'status', 'predicted_class']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent making predictions',
    ['model'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Confidence scores of predictions',
    ['model', 'predicted_class'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
)

HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percentage')
MEMORY_AVAILABLE_MB = Gauge('memory_available_mb', 'Available memory in MB')
DISK_USAGE = Gauge('disk_usage_percent', 'Disk usage percentage')

MODEL_LOADED = Gauge('model_loaded', 'Whether model is loaded (1) or not (0)')
MODEL_VERSION = Info('model_version', 'Model version information')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load model')
ACTIVE_MODELS = Gauge('active_models', 'Number of active models')

ERRORS_TOTAL = Counter(
    'errors_total',
    'Total number of errors',
    ['type', 'severity']
)

FEATURE_STATS = Gauge(
    'feature_statistics',
    'Statistics of input features',
    ['feature', 'statistic']
)


def simulate_predictions():
    classes = ['setosa', 'versicolor', 'virginica']
    
    for _ in range(random.randint(5, 15)):
        predicted_class = random.choice(classes)
        latency = random.uniform(0.001, 0.05)
        confidence = random.uniform(0.85, 1.0)
        
        PREDICTIONS_TOTAL.labels(
            model='iris-classifier',
            status='success',
            predicted_class=predicted_class
        ).inc()
        
        PREDICTION_LATENCY.labels(model='iris-classifier').observe(latency)
        PREDICTION_CONFIDENCE.labels(
            model='iris-classifier',
            predicted_class=predicted_class
        ).observe(confidence)
    
    if random.random() < 0.1:
        PREDICTIONS_TOTAL.labels(
            model='iris-classifier',
            status='error',
            predicted_class='unknown'
        ).inc()
        ERRORS_TOTAL.labels(type='prediction', severity='warning').inc()


def simulate_http_requests():
    endpoints = ['/predict', '/health', '/metrics', '/batch-predict']
    methods = ['GET', 'POST']
    statuses = ['200', '201', '400', '500']
    
    for _ in range(random.randint(10, 30)):
        endpoint = random.choice(endpoints)
        method = 'GET' if endpoint in ['/health', '/metrics'] else random.choice(methods)
        status = random.choices(statuses, weights=[0.85, 0.05, 0.07, 0.03])[0]
        duration = random.uniform(0.01, 0.3)
        
        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        HTTP_REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)


def update_system_metrics():
    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.percent)
        MEMORY_AVAILABLE_MB.set(memory.available / 1024 / 1024)
        
        disk = psutil.disk_usage('/')
        DISK_USAGE.set(disk.percent)
    except Exception as e:
        print(f"Error updating system metrics: {e}")


def update_model_metrics():
    MODEL_LOADED.set(1)
    ACTIVE_MODELS.set(1)
    MODEL_LOAD_TIME.set(1.85)
    
    MODEL_VERSION.info({
        'name': 'iris-classifier',
        'version': '1.0.0',
        'algorithm': 'RandomForest',
        'author': 'Rudy Peter Agung Chendra',
        'dicoding_id': 'M208D5Y1771',
        'training_date': '2025-12-20'
    })


def update_feature_statistics():
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    statistics = ['mean', 'std', 'min', 'max']
    
    for feature in features:
        for stat in statistics:
            value = random.uniform(0, 10)
            FEATURE_STATS.labels(feature=feature, statistic=stat).set(value)


def main():
    print("Prometheus Metrics Server")
    print("Rudy Peter Agung Chendra (M208D5Y1771)")
    print("Starting metrics server on port 8000...")
    
    update_model_metrics()
    start_http_server(8000)
    print("Metrics server started at http://localhost:8000/metrics")
    print("Generating simulated metrics...")
    print("Press Ctrl+C to stop")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            
            simulate_predictions()
            simulate_http_requests()
            update_system_metrics()
            update_feature_statistics()
            
            if iteration % 10 == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Metrics updated (iteration {iteration})")
            
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\nShutting down metrics server...")


if __name__ == "__main__":
    main()
