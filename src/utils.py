# Tree plotting and features imprtance

from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

def Plot_decision_tree(dt_model, feature_names, class_names, out_path="outputs/decision_tree.png", max_depth=3):
    plt.figure(figsize=(20,10))
    tree.plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True, max_depth=max_depth)
    plt.savefig(out_path)
    plt.close()

def Plot_feature_importances(model, feature_names, out_path="outputs/rf_feature_importances.png", top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8,6))
    plt.barh([feature_names[i] for i in indices[::-1]], importances[indices[::-1]])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
