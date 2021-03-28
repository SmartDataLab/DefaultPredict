import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
import pandas as pd
from feature.engine import pca_feature, lasso_feature, iv_feature
from feature.valid import feature_select_valid_model, evaluate
from sklearn.metrics import roc_curve, roc_auc_score, auc
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance


def pca_selecting(train_feature, train_label, test_feature, test_label):
    for i in tqdm(range(1, 10)):
        num_components = round((10 - i) / 10 * len(train_feature.columns))
        pca_train, pca_test, pca = pca_feature(
            train_feature, test_feature, n_components=num_components
        )
        valid_model = feature_select_valid_model()
        valid_model.fit(pca_train, train_label)
        valid_pred = valid_model.predict_proba(pca_test)[:, 1]
        valid_evaluation = evaluate(test_label, valid_pred)
        yield {
            "model": pca,
            "n_components": num_components,
            "features": ["com_%s" % i for i in range(num_components)],
            # using weight to show importance, gradually modify the best component we need
            "score": valid_evaluation,
            "evr": pca.explained_variance_ratio_
            # score based on the x-fold validate
        }


def rmse_cv(model, train_feature, train_label):
    rmse = np.sqrt(
        -cross_val_score(
            model, train_feature, train_label, scoring="neg_mean_squared_error", cv=3
        )
    )
    return rmse


def lasso_selecting(train_feature, train_label, alpha_base):
    for alpha in tqdm(range(1, 11)):
        model_lasso = LassoCV(alphas=[alpha * alpha_base]).fit(
            train_feature, train_label
        )
        non_zero_feature = [
            (column, model_lasso.coef_[i])
            for i, column in enumerate(train_feature.columns)
            if model_lasso.coef_[i] != 0.0
        ]
        corr_df = train_feature[[tuple_[0] for tuple_ in non_zero_feature]].corr()
        yield {
            "model": model_lasso,
            "alpha": model_lasso.alpha_,
            "features": non_zero_feature,
            "rmse": rmse_cv(model_lasso, train_feature, train_label).mean(),
            "corr_df": corr_df,
        }


def draw_pca_feature_selecting(pca_iterator, save_path):
    figure = plt.figure()
    n_components_list = []
    auc_list = []
    evr_list = []
    for one in pca_iterator:
        n_components_list.append(one["n_components"])
        auc_list.append(one["score"]["AUC"])
        evr_list.append(np.sum(one["evr"]))
    plt.plot(
        n_components_list,
        auc_list,
    )
    plt.plot(
        n_components_list,
        evr_list,
    )
    plt.legend(("AUC", "Explained Variance Ratio"), loc="upper right")
    plt.title("pca selecting precedure")
    plt.xlabel("n_components")
    plt.savefig(save_path)
    plt.close(figure)


def draw_lasso_feature_selecting(lasso_iterator, save_path):
    figure = plt.figure()
    n_feature_list = []
    alpha_list = []
    rmse_list = []
    corr_list = []
    for one in lasso_iterator:
        rmse_list.append(one["rmse"])
        alpha_list.append(one["alpha"])
        corr_list.append(one["corr_df"].mean())
        n_feature_list.append(len(one["features"]))
    ax1 = figure.add_subplot(111)
    ax1.plot(
        alpha_list,
        n_feature_list,
    )
    ax1.set_ylabel("n_feature")
    ax1.title("lasso selecting precedure")
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(alpha_list, rmse_list, "r")
    ax2.plot(alpha_list, corr_list, "g")
    ax2.set_ylabel("RMSE")
    ax2.set_xlabel("Same")
    plt.legend(("n_feature", "RMSE", "Corr"), loc="upper right")
    plt.xlabel("alpha")
    plt.savefig(save_path)
    plt.close(figure)


def draw_iv_feature_importance(iv_dict, save_path):
    pass


def plot_evaluation(y, prob_y, save_path_folder, method="", mode=""):
    plot_roc_curve(
        y, prob_y, "%s/%s:ROC_%s.png" % (save_path_folder, method, mode), mode=mode
    )
    cm = confusion_matrix(y, prob_y > 0.5)
    plot_confusion_matrix(cm, "%s/%s:cm(0.5)_%s.png" % (save_path_folder, method, mode))
    plot_ks_curve(y, prob_y, "%s/%s:ks_%s.png" % (save_path_folder, method, mode))
    plot_confusing_matrix_change(
        y, prob_y, "%s/%s:cm_change_%s.png" % (save_path_folder, method, mode)
    )


def plot_roc_curve(y, prob_y, save_path, mode=""):
    """
    plot roc curve
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    ----------------------------------
    plt object
    """
    figure = plt.figure()
    fpr, tpr, _ = roc_curve(y, prob_y)
    c_stats = roc_auc_score(y, prob_y)
    plt.plot([0, 1], [0, 1], "r--")
    plt.plot(fpr, tpr, label="ROC curve")
    s = "AUC = %.2f" % c_stats
    plt.text(0.6, 0.2, s, bbox=dict(facecolor="red", alpha=0.5))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve %s" % mode)  # ROC 曲线
    plt.legend(loc="best")
    plt.savefig(save_path)  # "figure/ROC_%s.png" % mode
    plt.close(figure)


def plot_confusion_matrix(
    cm,
    save_path,
    classes=[0, 1],
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    figure = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)  # 'figure/matrix(%s).png'%threshold
    plt.close(figure)


def plot_confusing_matrix_change(y, y_prob, save_path):
    plot_dict = {"thres": [], "TP": [], "FP": [], "FN": [], "TN": []}
    for i in range(21):
        threshold = i * 0.05
        y_prob_label = (y_prob >= threshold).astype(np.int)
        cm = confusion_matrix(y, y_prob_label)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        plot_dict["thres"].append(threshold)
        plot_dict["TP"].append(TP)
        plot_dict["FP"].append(FP)
        plot_dict["FN"].append(FN)
        plot_dict["TN"].append(TN)
    figure = plt.figure()
    plt.plot(plot_dict["thres"], plot_dict["TP"], label="TP")
    plt.plot(plot_dict["thres"], plot_dict["FP"], label="FP")
    plt.plot(plot_dict["thres"], plot_dict["FN"], label="FN")
    plt.plot(plot_dict["thres"], plot_dict["TN"], label="TN")
    plt.ylabel("count")
    plt.xlabel("threshold")
    plt.legend(loc="best")
    plt.savefig(save_path)  # "figure/confusing_change.png"
    plt.close(figure)


def cal_KS(target, score):
    """
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值 KS变化曲线
    """
    df = pd.DataFrame({"target": target, "score": score})
    total = df.groupby(["score"])["target"].count()
    bad = df.groupby(["score"])["target"].sum()
    all = pd.DataFrame({"total": total, "bad": bad})

    all["good"] = all["total"].to_numpy() - all["bad"].to_numpy()

    all["score_"] = all.index
    all = all.sort_values(by="score_", ascending=False)
    all.index = range(len(all))
    all["badCumRate"] = all["bad"].cumsum().to_numpy() / all["bad"].sum()
    all["goodCumRate"] = all["good"].cumsum().to_numpy() / all["good"].sum()
    all["KS"] = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return all[["badCumRate", "goodCumRate", "KS"]], all["KS"].max()


def plot_ks_curve(y, prob_y, save_path):
    KS_cur, KS = cal_KS(y, prob_y)
    KS_cur["accum_rate"] = KS_cur.index
    KS_cur["accum_rate"] = KS_cur["accum_rate"].to_numpy() / KS_cur.shape[0]
    lw = 2
    figure = plt.figure()
    # plt.figure(figsize=(5, 5))
    plt.plot(
        KS_cur["accum_rate"],
        KS_cur["badCumRate"],
        color="darkorange",
        lw=lw,
        label="KS_bad curve",
    )
    plt.plot(
        KS_cur["accum_rate"],
        KS_cur["goodCumRate"],
        color="navy",
        lw=lw,
        label="KS_good curve ",
    )
    plt.plot(KS_cur["accum_rate"], KS_cur["KS"], color="red", lw=lw, label="KS_curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Pro_of_default")
    plt.ylabel("Accumulate_Rate")
    plt.title("KS_CUR (KS=%0.2f) train data" % KS)
    plt.legend(loc="lower right")
    plt.savefig(save_path)  # "figure/KS_CUR_train.png"
    plt.close(figure)


def draw_xg_feature_importance(model, save_path):
    figure = plt.gcf()
    plot_importance(model)
    plt.savefig(save_path, figsize=(50, 40), dpi=1000)
    plt.close(figure)