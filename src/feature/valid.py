from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix
import json


def feature_select_valid_model():
    return LogisticRegression(
        C=0.05,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        multi_class="ovr",
        n_jobs=1,
        penalty="l2",
        random_state=None,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )


def evaluate(test_label, test_pred, threshold=0.5, save_path=None):
    mse = mean_squared_error(test_label, test_pred)
    er = 1 - precision_score(test_label, test_pred > threshold)
    R2 = r2_score(test_label, test_pred)
    auc = roc_auc_score(test_label, test_pred)
    cm = confusion_matrix(test_label, test_pred > threshold).tolist()
    res = {"MSE": mse, "er": er, "R2": R2, "AUC": auc, "CM": cm}
    if save_path:
        json.dump(res, open(save_path, "w"))
    return res
