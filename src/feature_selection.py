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

df.to_csv("../data/cleaned_all_data.csv", index=False)

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
import pickle

json.dump(interval_dict, open("../data/interval.json", "w"))

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
from visualization import EML

reload(EML)

EML.draw_pca_feature_selecting(
    pca_selector, save_path="../figure/pca_feature_selecting.png"
)
#%%
from visualization import EML

reload(EML)
lasso_selector = EML.lasso_selecting(
    train_feature, train_label, test_feature, test_label, alpha_base=0.001
)

EML.draw_lasso_feature_selecting(
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
    train_feature, test_feature, train_label, C=1e-3
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
