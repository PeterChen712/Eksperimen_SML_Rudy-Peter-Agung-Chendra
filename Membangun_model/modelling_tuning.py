import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "heart-disease-tuning"

def load_preprocessed_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    return X_train, X_test, y_train, y_test

def create_confusion_matrix_plot(y_true, y_pred, labels, save_path):
    import matplotlib
    matplotlib.use('Agg')
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

def create_feature_importance_plot(model, feature_names, save_path):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    return None

def train_with_manual_logging(X_train, X_test, y_train, y_test):
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    if dagshub_token:
        import dagshub
        dagshub.init(repo_owner='PeterChen712', 
                     repo_name='Eksperimen_SML_Rudy-Peter-Agung-Chendra', 
                     mlflow=True)
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'log_loss': log_loss(y_test, y_pred_proba),
        'cv_score_mean': grid_search.best_score_,
        'cv_score_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }
    
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    with mlflow.start_run():
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="heart-disease-rf")
        
        cm_path = create_confusion_matrix_plot(
            y_test, y_pred, 
            ['No Disease', 'Disease'],
            os.path.join(artifact_dir, 'confusion_matrix.png')
        )
        mlflow.log_artifact(cm_path)
        
        fi_path = create_feature_importance_plot(
            best_model,
            list(X_train.columns),
            os.path.join(artifact_dir, 'feature_importance.png')
        )
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        report = classification_report(y_test, y_pred, 
                                        target_names=['No Disease', 'Disease'],
                                        output_dict=True)
        report_path = os.path.join(artifact_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        gs_results = pd.DataFrame(grid_search.cv_results_)
        gs_path = os.path.join(artifact_dir, 'grid_search_results.csv')
        gs_results.to_csv(gs_path, index=False)
        mlflow.log_artifact(gs_path)
        
        model_summary = {
            'model_type': 'RandomForestClassifier',
            'best_params': best_params,
            'metrics': metrics,
            'feature_names': list(X_train.columns),
            'n_classes': 2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        summary_path = os.path.join(artifact_dir, 'model_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(model_summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path)
        
        scaler_path = "heart_preprocessing/scaler.pkl"
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path)
    
    return best_model, metrics

def main():
    train_path = "heart_preprocessing/heart_train_preprocessed.csv"
    test_path = "heart_preprocessing/heart_test_preprocessed.csv"
    
    X_train, X_test, y_train, y_test = load_preprocessed_data(train_path, test_path)
    train_with_manual_logging(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
