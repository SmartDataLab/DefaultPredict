#%%
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
from path import Path
from tqdm import tqdm
import json
from importlib import reload

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from logitboost import LogitBoost
from xgboost import XGBClassifier

from preprocess.transform import (
    get_total_data,
    label_build,
    vertical_nan_keep,
    horizontal_nan_keep,
    get_train_test,
    count_transform,
    chimerge,
    percent_intervals,
    box_transform,
    fill_nan_mean,
    get_contrast_data_from_df,
    get_balance_data_from_df,
    get_norm_feature,
)
from visualization.EDA import (
    draw_accumulation_plot,
    draw_boxplot,
    draw_density_plot,
    draw_corr_heat_map,
    draw_pandemic_contrast,
)
from visualization.EML import (
    pca_selecting,
    lasso_selecting,
    draw_lasso_feature_selecting,
    draw_pca_feature_selecting,
    draw_iv_feature_importance,
    plot_evaluation,
    draw_xg_feature_importance,
)
from feature.engine import pca_feature, lasso_feature, iv_feature
from feature.valid import feature_select_valid_model, evaluate
from models.LogisticVariants import TransferLogistic


#%%
# preprocess with the data input
# 1. build 0-1 label
# 2. drop the row and column with too many nan
# 3. fill the remaining nan with ffill method and mean fill
# 4. drop the centrailized column and will-not-use column
path_list = [
    "../data/正常.csv",
    "../data/逾期数据.csv",
    "../data/extra.csv",
]
df = get_total_data(path_list)
df = label_build(df)
draw_accumulation_plot(df, "../figure/累积贷款人数.png")
#%%
center_columns = [
    "user_blacklist_blacklist_name_with_idcard",
    "user_blacklist_blacklist_name_with_phone",
    "user_gray_has_report",
    "user_searched_history_by_day_d15_cnt_cc",
    "user_searched_history_by_day_d15_cnt_org_cc",
    "user_searched_history_by_day_d30_cnt_cc",
    "user_searched_history_by_day_d30_cnt_org_cc",
    "user_searched_history_by_day_d60_cnt_cc",
    "user_searched_history_by_day_d60_cnt_org_cc",
    "user_searched_history_by_day_d7_cnt_cc",
    "user_searched_history_by_day_d7_cnt_org_cc",
    "user_searched_history_by_day_d90_cnt_cc",
    "user_searched_history_by_day_d90_cnt_org_cc",
    "user_searched_history_by_day_m12_cnt_org_cc",
    "user_searched_history_by_day_m4_cnt_cc",
    "user_searched_history_by_day_m4_cnt_org_cc",
    "user_searched_history_by_day_m5_cnt_cc",
    "user_searched_history_by_day_m5_cnt_org_cc",
    "user_searched_history_by_day_m6_cnt_cc",
    "user_searched_history_by_day_m6_cnt_org_cc",
    "user_searched_history_by_day_m9_cnt_org_cc",
    "user_basic_user_idcard_valid",
]

unable_use_columns = [
    "user_basic_user_phone_city",
    "user_register_org_phone_num",
    "user_basic_user_region",
    "user_basic_user_city",
    "user_idcard_suspicion_idcard_applied_in_orgs",
    "user_phone_suspicion",
    "user_register_org_register_orgs_statistics",
    "user_idcard_suspicion_idcard_with_other_phones",
    "user_idcard_suspicion_idcard_with_other_names",
    "serialno",
    "user_gray_user_phone",
    "overduedays",
    "serialno.1",
    "LOANPUTOUT_DATE",
    "user_blacklist_blacklist_update_time_name_idcard",
    "contacts_number_statistic_cnt_be_all",
    "user_blacklist_blacklist_update_time_name_phone",
    "APPLYDATE",
    "contacts_gray_score_most_familiar_all",
    "user_blacklist_blacklist_category",
    "user_blacklist_blacklist_details",
    "consumer_label_phone_with_other_idcards",
    "OVERDUECORP",
]
df = df.drop(center_columns + unable_use_columns, axis=1)


df = df[vertical_nan_keep(df)]
df = df[horizontal_nan_keep(df)]
#%%
# convert date to number, str variables to number
df["user_searched_history_by_orgs"] = df["user_searched_history_by_orgs"].apply(
    lambda x: len(json.loads(x)) if type(x) == str else x
)
count_transform_columns = [
    "user_basic_user_province",
    "user_basic_user_phone_province",
    "user_basic_user_phone_operator",
]
for column in count_transform_columns:
    df[column] = count_transform(df[column])

#%%
df = df.fillna(method="ffill")
nan_count_dict = {column: np.sum(np.isnan(df[column])) for column in df.columns}
nan_column_list = [key for key, value in nan_count_dict.items() if value > 0]
df = fill_nan_mean(df, nan_column_list)
inf_count_dict = {column: np.sum(np.isinf(df[column])) for column in df.columns}
inf_column_list = [key for key, value in inf_count_dict.items() if value > 0]
inf_column_list


#%%
feature_columns = list(df.columns)
feature_columns.pop(feature_columns.index("target"))
feature_columns.pop(feature_columns.index("dataset"))
label_column = "target"
train_feature, test_feature, train_label, test_label = get_train_test(
    df, feature_columns, label_column
)


# speed up
need_chimerge_columns = []
need_percent_columns = []
for column in tqdm(train_feature.columns):
    column_set = set(train_feature[column].unique())
    if len(column_set) > 40:
        if len(column_set) < 2000:
            need_chimerge_columns.append(column)
        else:
            need_percent_columns.append(column)
#%%
# feature engineering
# 1. get the intervals from both chisquare
# 2. transform the feature into categorical variables

interval_dict = {}
for column in tqdm(need_chimerge_columns):
    intervals = chimerge(
        data=train_feature,
        attr=column,
        label=train_label,
        downsamplecount=100,
        max_intervals=6,
    )
    interval_dict[column] = intervals
#%%

for column in tqdm(need_percent_columns):
    intervals = percent_intervals(
        data=train_feature,
        attr=column,
        label=train_label,
        downsamplecount=1000,
        max_intervals=6,
    )
    interval_dict[column] = intervals

#%%
except_count = 0
for column, intervals in tqdm(interval_dict.items()):
    try:
        train_feature[column] = train_feature[column].apply(
            lambda x: box_transform(x, intervals)
        )
        test_feature[column] = test_feature[column].apply(
            lambda x: box_transform(x, intervals)
        )
    except Exception as e:
        print(column, e)
        except_count += 1
print(except_count)
#%%
train_feature, test_feature = get_norm_feature(train_feature, test_feature)
#%%
# feature selection
# 1. draw feature importance during feature selection
# 2. get prepared for training data from pca, lasso and iv
pca_selector = pca_selecting(
    train_feature,
    train_label,
    test_feature,
    test_label,
)

#%%
draw_pca_feature_selecting(
    pca_selector, save_path="../figure/pca_feature_selecting.png"
)
#%%
lasso_selector = lasso_selecting(train_feature, train_label, alpha_base=0.001)


draw_lasso_feature_selecting(
    lasso_selector, save_path="../figure/lasso_feature_selecting.png"
)
#%%


pca_train, pca_test, _ = pca_feature(train_feature, test_feature, n_components=20)
draw_density_plot(
    pca_train, pca_train.columns, save_path="../figure/density.png", xlim=[-2, 2]
)
json.dump(list(pca_train.columns), open("../data/feature_columns(PCA).json", "w"))

#%%
lasso_train, lasso_test, lasso = lasso_feature(
    train_feature, test_feature, train_label, alpha=0.007
)
json.dump(list(lasso_train.columns), open("../data/feature_columns(LASSO).json", "w"))
#%%

iv_train, iv_test, iv_feature, iv_dict = iv_feature(
    train_feature,
    test_feature,
    train_label,
    interval_dict,
    cr_threshold=0.8,
    n_feature=20,
)
draw_iv_feature_importance(iv_dict, save_path="../figure/iv_feature_importance.png")
json.dump(list(iv_train.columns), open("../data/feature_columns(IV).json", "w"))
#%%
# contrast plot before and after deleting the highly-correlate variables
raw_iv_feature = sorted(
    [(column, detail_dict["iv"]) for column, detail_dict in iv_dict.items()],
    key=lambda x: x[1],
    reverse=True,
)
raw_iv_feature = [x[0] for x in raw_iv_feature[:20]]
draw_corr_heat_map(
    train_feature[raw_iv_feature], "../figure/correlation_heat_map(before).png"
)
draw_corr_heat_map(
    train_feature[iv_feature], "../figure/correlation_heat_map(after).png"
)
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


#%%
# using logistic to valid different feature selection methods


pca_logistic = LogisticRegression()  # feature_select_valid_model()
pca_logistic.fit(pca_train, train_label)
pca_pred = pca_logistic.predict_proba(pca_test)[:, 1]
pca_evaluation = evaluate(
    test_label, pca_pred, save_path="../data/lr(pca)_evaluation.json"
)

log_reg = sm.Logit(train_label, pca_train).fit()
summary = log_reg.summary()
with open("../data/logistic_result(PCA).csv", "w") as f:
    f.write(summary.as_csv())

lasso_logistic = LogisticRegression()  # feature_select_valid_model()
lasso_logistic.fit(lasso_train, train_label)
lasso_pred = lasso_logistic.predict_proba(lasso_test)[:, 1]
lasso_evaluation = evaluate(
    test_label, lasso_pred, save_path="../data/lr(lasso)_evaluation.json"
)

log_reg = sm.Logit(train_label, lasso_train).fit()
summary = log_reg.summary()
with open("../data/logistic_result(LASSO).csv", "w") as f:
    f.write(summary.as_csv())

iv_logistic = LogisticRegression()  # feature_select_valid_model()
iv_logistic.fit(iv_train, train_label)
iv_pred = iv_logistic.predict_proba(iv_test)[:, 1]
iv_evaluation = evaluate(
    test_label, iv_pred, save_path="../data/lr(iv)_evaluation.json"
)

log_reg = sm.Logit(train_label, iv_train).fit()
summary = log_reg.summary()
with open("../data/logistic_result(IV).csv", "w") as f:
    f.write(summary.as_csv())

#%%

# balance dataset testing
key_feature = list(lasso_train.columns)
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
key_feature = list(lasso_train.columns)

best_feature_pred = lasso_pred
plot_evaluation(test_label, best_feature_pred, "../figure", method="Logistic")

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

# TODO(sujinhua): remove the valid
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

# TODO(sujinhua): add XGboost - additive learning
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
