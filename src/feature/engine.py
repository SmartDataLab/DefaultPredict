import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


def pca_feature(train_feature, test_feature, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(train_feature)
    train_feature_ = pd.DataFrame(
        pca.transform(train_feature),
        columns=["PCA_%s" % (i + 1) for i in range(n_components)],
    )
    test_feature_ = pd.DataFrame(
        pca.transform(test_feature),
        columns=["PCA_%s" % (i + 1) for i in range(n_components)],
    )
    return train_feature_, test_feature_, pca


def lasso_feature(train_feature, test_feature, train_label, alpha):
    model_lasso = LassoCV(alphas=[alpha]).fit(train_feature, train_label)

    coef = pd.Series(np.abs(model_lasso.coef_), index=train_feature.columns)
    feature_columns = list(coef[coef != 0].index)
    return train_feature[feature_columns], test_feature[feature_columns], model_lasso


def iv_feature(
    train_feature,
    test_feature,
    train_label,
    interval_dict,
    cr_threshold,
    n_feature=None,
):
    except_columns = []
    iv_dict = {}
    for column in tqdm(train_feature.columns):
        iv, IV, WOE, N_0_group, N_1_group = CalcIV(train_feature[column], train_label)
        try:
            intervals = interval_dict[column]
        except:
            except_columns.append(column)
            intervals = np.unique(train_feature[column])

        iv_dict[column] = {"iv": iv}
        if len(intervals) > len(N_1_group):
            intervals = intervals[-len(N_1_group) :]
        for i in range(len(intervals)):
            iv_dict[column]["Bin%s" % i] = {
                "range": intervals[i],
                "positive": N_1_group[i],
                "negative": N_0_group[i],
                "WOE": WOE[i],
                "IV": IV[i],
            }

    print("there is %s columns not in interval_dict " % len(except_columns))
    print(except_columns)
    # there must be some calculation for this xxxx
    pd.DataFrame(iv_dict[column]).to_csv("../data/Bins/%s.csv" % column)
    iv_df = pd.DataFrame(
        [{"name": key, "IV": value["iv"]} for key, value in iv_dict.items()]
    )
    iv_df = iv_df.sort_values("IV", ascending=False)
    cor_df = train_feature.corr()
    delete_high_corr_columns = []
    for column in iv_df["name"]:
        if column not in delete_high_corr_columns:
            for column2 in iv_df["name"]:
                if (
                    cor_df.loc[column, column2] > 0.8
                    and column != column2
                    and column2 not in delete_high_corr_columns
                ):
                    delete_high_corr_columns.append(column2)
    key_feature = []
    for name in iv_df["name"]:
        if len(key_feature) >= 20:
            break
        if name not in delete_high_corr_columns:
            key_feature.append(name)
    return train_feature[key_feature], test_feature[key_feature], key_feature, iv_dict


def CalcIV(Xvar, Yvar):
    Xvar = Xvar.fillna(Xvar.mean())
    x_unique = np.unique(Xvar)
    N_0 = np.sum(Yvar == 0)
    N_1 = np.sum(Yvar == 1)
    N_0_group = np.zeros(x_unique.shape)
    N_1_group = np.zeros(x_unique.shape)
    for i in range(len(x_unique)):
        N_0_group[i] = Yvar[(Xvar == x_unique[i]) & (Yvar == 0)].count() + 1
        N_1_group[i] = Yvar[(Xvar == x_unique[i]) & (Yvar == 1)].count() + 1
    WOE = np.log((N_0_group / N_0) / (N_1_group / N_1))
    IV = (N_0_group / N_0 - N_1_group / N_1) * WOE
    iv = np.sum(IV)
    return iv, IV, WOE, N_0_group, N_1_group
