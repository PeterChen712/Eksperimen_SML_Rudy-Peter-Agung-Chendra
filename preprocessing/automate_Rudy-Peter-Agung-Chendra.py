import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_raw_data(file_path='heart.csv'):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df_processed = df.copy()
    df_processed = df_processed.replace('?', np.nan)
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed = df_processed.fillna(df_processed.median())
    df_processed = df_processed.drop_duplicates()
    print(f"Data shape after preprocessing: {df_processed.shape}")
    print(f"Missing values: {df_processed.isnull().sum().sum()}")
    return df_processed

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='heart_preprocessing'):
    os.makedirs(output_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test
    train_path = os.path.join(output_dir, 'heart_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'heart_test_preprocessed.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Preprocessed data saved to {output_dir}/")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    return train_path, test_path

def run_preprocessing(input_file='heart.csv', output_dir='heart_preprocessing'):
    df = load_raw_data(input_file)
    df_cleaned = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_cleaned)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    train_path, test_path = save_preprocessed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, output_dir
    )
    return train_path, test_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess Heart Disease dataset')
    parser.add_argument('--input', type=str, default='heart.csv')
    parser.add_argument('--output', type=str, default='heart_preprocessing')
    args = parser.parse_args()
    train_path, test_path = run_preprocessing(args.input, args.output)
    print(f"Preprocessing complete!")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
