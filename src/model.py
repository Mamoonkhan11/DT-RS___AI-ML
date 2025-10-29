# Train and store models

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def Train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt

def Train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def Save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
