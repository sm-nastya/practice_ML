import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from typing import List
import joblib
import os
import sys

def train_model():
    X = np.load('X_data.npy', allow_pickle=True)
    y = np.load('y_data.npy', allow_pickle=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    os.makedirs('resources', exist_ok=True)
    joblib.dump(model, 'resources/salary_model.pkl')

def predict_salaries(X_path: str) -> List[float]:
    model = joblib.load('resources/salary_model.pkl')
    X_data = np.load(X_path, allow_pickle=True)
    salaries = model.predict(X_data)
    return salaries.to_list()

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Enter the path to hh.csv dataset")
        exit(0)

    X_path = sys.argv[1]
    train_model()
    salaries = predict_salaries(X_path)
    for item in salaries:
        print(item)
