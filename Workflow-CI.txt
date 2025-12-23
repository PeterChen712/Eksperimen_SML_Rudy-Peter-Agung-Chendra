# Pipeline CI/CD MLflow

Rudy Peter Agung Chendra (M208D5Y1771)

Pipeline otomatis untuk training dan deployment model machine learning menggunakan MLflow dan GitHub Actions.

## Deskripsi

Repository ini berisi infrastruktur CI/CD untuk automated model training dan deployment. Pipeline menangani training model, manajemen artifact, dan publishing Docker image.

## Struktur

```
MLProject/
├── MLProject              # Konfigurasi MLflow project
├── conda.yaml            # Spesifikasi environment conda
├── modelling.py          # Script training
└── iris_preprocessing/   # Data yang sudah diproses
```

## Penggunaan

### Training Lokal

```bash
cd MLProject
python modelling.py --n_estimators 100 --max_depth 5
```

### MLflow Project

```bash
mlflow run . -P n_estimators=100 -P max_depth=5
```

### GitHub Actions

Workflow trigger otomatis saat push ke main branch. Manual trigger juga tersedia melalui workflow_dispatch.

Parameter:
- n_estimators: Jumlah pohon di random forest (default: 100)
- max_depth: Kedalaman maksimum pohon (default: 5)

## Pipeline CI

Pipeline terdiri dari tiga job utama:

1. Training - Menjalankan script training model
2. Artifact Upload - Menyimpan model artifacts ke repository
3. Docker Build - Membuat dan push Docker image (memerlukan secrets)

## Requirements

- Python 3.12
- Dependencies: pandas, numpy, scikit-learn, mlflow, joblib

## Konfigurasi

Untuk publishing Docker Hub, konfigurasi repository secrets:
- DOCKER_USERNAME
- DOCKER_PASSWORD

## Repository Terkait

Eksperimen utama: https://github.com/PeterChen712/Eksperimen_SML_Rudy-Peter-Agung-Chendra
