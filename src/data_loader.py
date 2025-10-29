# Load data from the Data folder

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def Load_data(file_path, target_col):
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')  
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
