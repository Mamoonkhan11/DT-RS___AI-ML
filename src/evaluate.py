# Metrics and Cv configuration

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

def Evaluate_model(model, X_test, y_test, outputs_prefix="Outputs/model"):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except:
        y_proba = None

    # Confusion matrix setup
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig(f"{outputs_prefix}_confusion_matrix.png"); plt.close()

    # Classification report setup
    report = classification_report(y_test, y_pred)
    with open(f"{outputs_prefix}_report.txt","w") as f:
        f.write(report)

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        fpr,tpr,_ = roc_curve(y_test, y_proba)
        plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.legend(); plt.savefig(f"{outputs_prefix}_roc.png"); plt.close()

    return report

def Cross_validate_model(estimator, X, y, cv=5, scoring='accuracy'):
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores
