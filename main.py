# Main script to load data, train models, evaluate them, and save outputs.

from src.data_loader import Load_data
from src.model import Train_decision_tree, Train_random_forest, Save_model
from src.evaluate import Evaluate_model
from src.utils import Plot_decision_tree, Plot_feature_importances

def main():
    print("\n Loading dataset...\n")
    X_train, X_test, y_train, y_test = Load_data(
        file_path="Data/heart.csv", target_col="target"
    )

    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    class_names = ["No Disease", "Disease"]

    print("\n Training Decision Tree model...\n")
    dt_model = Train_decision_tree(X_train, y_train, max_depth=4)
    Save_model(dt_model, "Outputs/decision_tree_model.pkl")

    print("\n Evaluating Decision Tree model...\n")
    Evaluate_model(dt_model, X_test, y_test, outputs_prefix="Outputs/decision_tree")
    Plot_decision_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_names,
        out_path="Outputs/decision_tree.png"
    )

    print("\n Training Random Forest model...\n")
    rf_model = Train_random_forest(X_train, y_train, n_estimators=200)
    Save_model(rf_model, "Outputs/random_forest_model.pkl")

    print("\n Evaluating Random Forest model...\n")
    Evaluate_model(rf_model, X_test, y_test, outputs_prefix="Outputs/random_forest")
    Plot_feature_importances(
        rf_model,
        feature_names=feature_names,
        out_path="Outputs/rf_feature_importances.png"
    )

    print("\n All tasks completed successfully! Check the outputs folder for results.\n")

if __name__ == "__main__":
    main()
