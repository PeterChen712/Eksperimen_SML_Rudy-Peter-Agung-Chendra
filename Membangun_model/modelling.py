import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import argparse

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

def prepare_data():
    train_path = "heart_preprocessing/heart_train_preprocessed.csv"
    test_path = "heart_preprocessing/heart_test_preprocessed.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    return X_train, X_test, y_train, y_test

def train_model(n_estimators=100, max_depth=5, min_samples_split=2):
    X_train, X_test, y_train, y_test = prepare_data()
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        dagshub_token = os.getenv('DAGSHUB_TOKEN') or os.getenv('DAGSHUB_USER_TOKEN')
        if dagshub_token:
            import dagshub
            dagshub.init(
                repo_owner='PeterChen712',
                repo_name='Eksperimen_SML_Rudy-Peter-Agung-Chendra',
                mlflow=True
            )
        else:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    mlflow.set_experiment("heart-disease-classification")
    
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", 42)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="heart-disease-classifier"
        )
        
        return model, {"accuracy": accuracy, "f1_score": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_samples_split', type=int, default=2)
    args = parser.parse_args()
    
    train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )

if __name__ == "__main__":
    main()
