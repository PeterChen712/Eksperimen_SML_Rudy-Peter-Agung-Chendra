import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

train_df = pd.read_csv('heart_preprocessing/heart_train_preprocessed.csv')
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

joblib.dump(model, 'heart_preprocessing/model.pkl')
print("Model saved to heart_preprocessing/model.pkl")
