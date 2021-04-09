#%%
import numpy as np
import pandas as pd
import json
from importlib import reload
# import thirdparty

#%%
from preprocess.transform import get_balance_data_from_df
from visualization.EML import plot_evaluation, draw_xg_feature_importance
from feature.valid import evaluate
from visualization.EDA import draw_boxplot, draw_pandemic_contrast
from logitboost import LogitBoost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import statsmodels.api as sm
from models.LogisticVariants import TransferLogistic

# %%
# to prove the heterogeneity
# 1. test the above and need to adjust the lasso so that it will excess the iv method
# 2. add more model like xgboost
# 3. save all the needed results
# previous best feature by iv method ↓
# key_feature = [
#     "contacts_number_statistic_pct_black_ratio",
#     "contacts_number_statistic_pct_cnt_to_black",
#     "contacts_number_statistic_pct_cnt_be_black",
#     "contacts_query_org_cnt_3",
#     "contacts_query_be_query_cnt_05",
#     "contacts_gray_score_most_familiar_be_all",
#     "contacts_rfm_call_cnt_to_applied",
#     "user_searched_history_by_day_d30_pct_cnt_org_cf",
#     "LOANTERM",
#     "contacts_gray_score_min",
#     "user_searched_history_by_day_d15_pct_cnt_org_cash",
#     "contacts_number_statistic_pct_router_ratio",
#     "contacts_query_to_query_cnt_9",
#     "user_searched_history_by_day_m4_cnt_org_cf",
#     "user_searched_history_by_day_d30_pct_cnt_org_cash",
#     "user_searched_history_by_day_d7_pct_cnt_org_cc",
#     "contacts_query_to_query_cnt_05",
#     "user_searched_history_by_day_m4_pct_cnt_org_all",
#     "user_searched_history_by_day_m4_pct_cnt_org_cash",
#     "user_searched_history_by_day_d90_pct_cnt_all",
# ]
key_feature = json.load(open("../data/feature_columns(LASSO).json"))
df = pd.read_csv("../data/cleaned_all_data.csv")
interval_dict = json.load(open("../data/interval.json"))

#%%

# get df and interval_dict and get key_feature


# TODO: new method to write the cost function curve

#%%
# balance dataset testing
for ratio in [0.2, 0.4, 0.5, 0.6, 0.8]:
    (
        feature_train_balance,
        label_train_balance,
        feature_test_balance,
        label_test_balance,
    ) = get_balance_data_from_df(df, interval_dict, key_feature, ratio)
    rf_balance = RandomForestClassifier(max_depth=4, random_state=0)
    rf_balance.fit(feature_train_balance, label_train_balance)
    rf_pred_balance = rf_balance.predict_proba(feature_test_balance)[:, 1]
    rf_evaluation = evaluate(
        label_test_balance,
        rf_pred_balance,
        save_path="../data/rf_balance(%s)_evaluation.json" % ratio,
    )
    plot_evaluation(
        label_test_balance, rf_pred_balance, "../figure", method="RF_balance_%s" % ratio
    )
# %%


#%%
# Other classification models


(
    feature_train_balance,
    label_train_balance,
    feature_test_balance,
    label_test_balance,
) = get_balance_data_from_df(df, interval_dict, key_feature, 0.5)

lr = LogisticRegression(
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

lr.fit(feature_train_balance, label_train_balance)
lr.fit(feature_train_balance, label_train_balance)
lr_pred = lr.predict_proba(feature_test_balance)[:, 1]
lr_evaluation = evaluate(
    label_test_balance, lr_pred, save_path="../data/lr_evaluation.json"
)
plot_evaluation(label_test_balance, lr_pred, "../figure", method="LR")

# show the detail of the logistic model

#%%
svm = SVC(C=1.0, probability=True)
svm_index = list(np.random.choice(list(feature_train_balance.index), 10000))
svm.fit(feature_train_balance.loc[svm_index, :], label_train_balance[svm_index])
svm_pred = svm.predict_proba(feature_test_balance)[:, 1]
svm_evaluation = evaluate(
    label_test_balance, svm_pred, save_path="../data/svm_evaluation.json"
)
plot_evaluation(label_test_balance, svm_pred, "../figure", method="SVM")
#%%
rf = RandomForestClassifier(max_depth=4, random_state=0)
rf.fit(feature_train_balance, label_train_balance)
rf_pred = rf.predict_proba(feature_test_balance)[:, 1]
rf_evaluation = evaluate(
    label_test_balance, rf_pred, save_path="../data/rf_evaluation.json"
)
plot_evaluation(label_test_balance, rf_pred, "../figure", method="RF")

#%%
from feature import valid

reload(valid)
xg = XGBClassifier(
    learning_rate=0.01,
    n_estimators=10,  # 树的个数-10棵树建立xgboost
    max_depth=4,  # 树的深度
    min_child_weight=1,  # 叶子节点最小权重
    gamma=0.0,  # 惩罚项中叶子结点个数前的参数
    subsample=1,  # 所有样本建立决策树
    colsample_btree=1,  # 所有特征建立决策树
    scale_pos_weight=1,  # 解决样本个数不平衡的问题
    random_state=27,  # 随机数
    slient=0,
)
xg.fit(feature_train_balance, label_train_balance)
xg_pred = xg.predict_proba(feature_test_balance)[:, 1]
xg_evaluation = valid.evaluate(
    label_test_balance, xg_pred, save_path="../data/xg_evaluation.json"
)
plot_evaluation(label_test_balance, xg_pred, "../figure", method="XG")
#%%

lb = LogitBoost(n_estimators=200, random_state=0)  # base_estimator=LogisticRegression()
lb.fit(feature_train_balance, label_train_balance)
lb_pred = lb.predict_proba(feature_test_balance)[:, 1]
lb_evaluation = evaluate(
    label_test_balance, lb_pred, save_path="../data/lb_evaluation.json"
)
plot_evaluation(label_test_balance, lb_pred, "../figure", method="LB")
#%%
from feature import valid

#%%
# Auto-tunan_column_listne model for pandemic
# 1. XGboost
# 2. XGboost - additive learning
# 3. LogisticBoosting - additive learning
# 4. dummy Logistic
# 5. Transfer Logistic
from preprocess import transform

reload(transform)
params = {"objective": "reg:linear", "verbose": False}
best_model = XGBClassifier(
    learning_rate=0.01,
    n_estimators=10,  # 树的个数-10棵树建立xgboost
    max_depth=4,  # 树的深度
    min_child_weight=1,  # 叶子节点最小权重
    gamma=0.0,  # 惩罚项中叶子结点个数前的参数
    subsample=1,  # 所有样本建立决策树
    colsample_btree=1,  # 所有特征建立决策树
    scale_pos_weight=1,  # 解决样本个数不平衡的问题
    random_state=27,  # 随机数
    slient=0,
)
(
    train_feature_2019,
    train_label_2019,
    test_feature_2020,
    test_label_2020,
) = transform.get_contrast_data_from_df(df, interval_dict, key_feature)

draw_boxplot(train_feature_2019, test_feature_2020, "../figure/箱线对比图.png")
best_model.fit(train_feature_2019, train_label_2019)
xg_2020_pred = best_model.predict_proba(test_feature_2020)[:, 1]
xg_2020_evaluation = valid.evaluate(
    test_label_2020, xg_2020_pred, save_path="../data/xg(2020)_evaluation.json"
)
plot_evaluation(test_label_2020, xg_2020_pred, "../figure", method="XG_2020")

#%%
# additive learning for xgboost
import xgboost as xgb

glimse_index = list(
    np.random.choice(list(test_feature_2020.index), 1000, replace=False)
)
test_index = list(set(test_feature_2020.index) - set(glimse_index))
params = best_model.get_xgb_params()
xg_2020_train = xgb.DMatrix(
    test_feature_2020.loc[glimse_index, :], label=test_label_2020[glimse_index]
)
xg_2020_test = xgb.DMatrix(
    test_feature_2020.loc[test_index, :], label=test_label_2020[test_index]
)
best_model.save_model("../data/xg_2019.model")
additive_xg = xgb.train(params, xg_2020_train, 5, xgb_model="../data/xg_2019.model")
additive_xg_pred = additive_xg.predict(xg_2020_test)
additive_xg_evaluation = valid.evaluate(
    test_label_2020[test_index],
    additive_xg_pred,
    save_path="../data/additive_xg(2020)_evaluation.json",
)
plot_evaluation(
    test_label_2020[test_index],
    additive_xg_pred,
    "../figure",
    method="additive_XG_2020",
)
#%%


# test p-values
lr_base = LogisticRegression(C=1e30).fit(train_feature_2019, train_label_2019)
tf = TransferLogistic(lr_base)
tf.fit(test_feature_2020.loc[glimse_index, :], test_label_2020[glimse_index])
tf_2020_pred = tf.predict_proba(test_feature_2020.loc[test_index, :])[:, 1]
tf_2020_evaluation = evaluate(
    test_label_2020[test_index],
    tf_2020_pred,
    save_path="../data/tf(2020)_evaluation.json",
)
plot_evaluation(
    test_label_2020[test_index], tf_2020_pred, "../figure", method="TF_2020"
)

#%%

draw_pandemic_contrast(
    best_model,
    tf,
    train_feature_2019,
    test_feature_2020.loc[test_index, :],
    "../figure/all_year_proba_curve.png",
)

#%%

draw_xg_feature_importance(xg, save_path="../figure/feature_importance(xgboost).png")
# %%

from sklearn.cluster import KMeans
import numpy as np
from models.ClusterDividePredict import ClusterPredictor
from models import ClusterDividePredict

reload(ClusterDividePredict)
ClusterPredictor = ClusterDividePredict.ClusterPredictor
kmeans = KMeans(n_clusters=5, random_state=0)
predict_models = [
    LogisticRegression(
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
    for _ in range(20)
]
cluster_predictor = ClusterPredictor(kmeans, predict_models)
cluster_predictor.fit(
    pd.concat(
        [train_feature_2019, test_feature_2020.loc[glimse_index, :]], ignore_index=True
    ),
    pd.concat([train_label_2019, test_label_2020[glimse_index]], ignore_index=True),
)
cp_2020_pred = cluster_predictor.predict_proba(test_feature_2020.loc[test_index, :])[
    :, 1
]
cp_2020_evaluation = evaluate(
    test_label_2020[test_index],
    cp_2020_pred,
    save_path="../data/cp(2020)_evaluation.json",
)
plot_evaluation(
    test_label_2020[test_index], cp_2020_pred, "../figure", method="CP_2020"
)

# %%
from models.CascadePredict import CascadePredictor

CaP = CascadePredictor(
    [RandomForestClassifier() for i in range(2)], lambda x: x < 0.7, 0.0
)
CaP.fit_list(
    [train_feature_2019, test_feature_2020.loc[glimse_index, :]],
    [train_label_2019, test_label_2020[glimse_index]],
)
cap_2020_pred = CaP.predict_proba(test_feature_2020.loc[test_index, :])[:,1]
cap_2020_evaluation = evaluate(
    test_label_2020[test_index],
    cap_2020_pred,
    save_path="../data/cap(2020)_evaluation.json",
)
plot_evaluation(
    test_label_2020[test_index], cap_2020_pred, "../figure", method="CAP_2020"
)