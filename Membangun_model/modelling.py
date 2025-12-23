import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import argparse

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
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("heart-disease-classification")
    
    try:
        mlflow.sklearn.autolog()
    except Exception as e:
        print(f"Warning: autolog error ({e}), trying with disable=True first")
        try:
            mlflow.sklearn.autolog(disable=True)
            mlflow.sklearn.autolog()
        except:
            pass
    
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model trained with accuracy: {score:.4f}")
    return model

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
