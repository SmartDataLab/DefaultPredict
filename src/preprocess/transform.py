import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Normalizer


def get_total_data(path_list):
    def read_data(path, i):
        df = pd.read_csv(path)
        df["dataset"] = i
        return df

    return pd.concat([read_data(path, i) for i, path in enumerate(path_list)])


def get_norm_feature(train_feature, test_feature):
    for column in tqdm(train_feature.columns):
        scaler = Normalizer().fit(train_feature[[column]])  # fit does nothing.
        train_feature[column] = scaler.transform(train_feature[[column]])[:, 0]
        test_feature[column] = scaler.transform(test_feature[[column]])[:, 0]
    return train_feature, test_feature


def get_balance_data_from_df(
    df, interval_dict, key_feature, ratio, label_column="target"
):
    n_postive_sample = round(len(df) * ratio)
    n_negative_sample = len(df) - n_postive_sample
    balance_data = pd.concat(
        [
            df[df["target"] == 0].sample(n_negative_sample, replace=True),
            df[df["target"] == 1].sample(n_postive_sample, replace=True),
        ]
    ).reset_index(drop=True)
    train_feature, test_feature, train_label, test_label = get_train_test(
        balance_data, key_feature, label_column
    )

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
    train_feature, test_feature = get_norm_feature(train_feature, test_feature)
    return train_feature, train_label, test_feature, test_label


def get_contrast_data_from_df(df, interval_dict, key_feature, label_column="target"):
    data_2019, data_2020 = (
        df[df["dataset"] != 2].reset_index(drop=True),
        df[df["dataset"] == 2].reset_index(drop=True),
    )
    train_feature_2019, train_label_2019 = (
        data_2019[key_feature],
        data_2019[label_column],
    )
    test_feature_2020, test_label_2020 = data_2020[key_feature], data_2020[label_column]
    for column, intervals in tqdm(interval_dict.items()):
        try:
            train_feature_2019[column] = train_feature_2019[column].apply(
                lambda x: box_transform(x, intervals)
            )
            test_feature_2020[column] = test_feature_2020[column].apply(
                lambda x: box_transform(x, intervals)
            )
        except Exception as e:
            print(column, e)

    train_feature_2019, test_feature_2020 = get_norm_feature(
        train_feature_2019, test_feature_2020
    )
    return train_feature_2019, train_label_2019, test_feature_2020, test_label_2020


def label_build(df):
    df_ = df.copy()
    df_["target"] = df_["overduedays"] > 0
    return df_


def vertical_nan_keep(df, keep_ratio=0.7):
    na_stat_dict = {}
    for column in df.columns:
        count = df[column].isna().sum()
        na_stat_dict[column] = count
    all_count = sum(na_stat_dict.values())
    return [
        key for key, value in na_stat_dict.items() if 1 - value / all_count > keep_ratio
    ]


def horizontal_nan_keep(df, keep_ratio=0.7):
    return df.apply(
        lambda x: True if 1 - x.isna().sum() / len(x) > keep_ratio else False, axis=1
    ).tolist()


def get_train_test(df, feature_columns, label_column):
    train_feature, test_feature, train_label, test_label = train_test_split(
        df[feature_columns], df[label_column], test_size=0.25, random_state=7
    )
    train_feature, test_feature = (
        train_feature.reset_index(drop=True),
        test_feature.reset_index(drop=True),
    )
    train_label, test_label = (
        train_label.reset_index(drop=True),
        test_label.reset_index(drop=True),
    )
    return train_feature, test_feature, train_label, test_label


def count_transform(series):
    unique = set(series)
    count_dict = {name: 0 for name in unique}
    for name in series:
        count_dict[name] += 1
    high2low = sorted(
        [(key, value) for key, value in count_dict.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    map_dict = {tuple_[0]: i for i, tuple_ in enumerate(high2low)}
    new_series = series.apply(lambda x: map_dict[x])
    return new_series


def chimerge(data, attr, label, max_intervals, downsamplecount=None):
    if type(label) != str:
        data = data.copy()
        data["target"] = label
        label = "target"
    if downsamplecount != None:
        data = data.sample(downsamplecount)
    distinct_vals = sorted(set(data[attr]))  # Sort the distinct values
    labels = sorted(set(data[label]))  # Get all possible labels
    empty_count = {l: 0 for l in labels}  # A helper function for padding the Counter()
    intervals = [
        [distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))
    ]  # Initialize the intervals for each attribute
    while len(intervals) > max_intervals:  # While loop
        chi = []
        for i in range(len(intervals) - 1):
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array(
                [v for i, v in {**empty_count, **Counter(obs0[label])}.items()]
            )
            count_1 = np.array(
                [v for i, v in {**empty_count, **Counter(obs1[label])}.items()]
            )
            count_total = count_0 + count_1
            expected_0 = count_total * sum(count_0) / total
            expected_1 = count_total * sum(count_1) / total
            chi_ = (count_0 - expected_0) ** 2 / expected_0 + (
                count_1 - expected_1
            ) ** 2 / expected_1
            chi_ = np.nan_to_num(chi_)  # Deal with the zero counts
            chi.append(sum(chi_))  # Finally do the summation for Chi2
        min_chi = min(chi)  # Find the minimal Chi2 for current iteration
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i  # Find the index of the interval to be merged
                break
        new_intervals = []  # Prepare for the merged new data array
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done:  # Merge the intervals
                t = intervals[i] + intervals[i + 1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        intervals = new_intervals
    return intervals


def box_transform(x, intervals):
    if x < intervals[0][0]:
        return -1
    for i, interval in enumerate(intervals):
        if x >= interval[0] and x <= interval[1]:
            return i
    return len(intervals)


def percent_intervals(data, attr, label, max_intervals, downsamplecount=None):
    if type(label) != str:
        data = data.copy()
        data["target"] = label
        label = "target"
    if downsamplecount != None:
        data = data.sample(downsamplecount)
    sort_vals = sorted(data[attr])  # Sort the distinct values
    intervals = [
        [
            sort_vals[int(len(data) * i / max_intervals)],
            sort_vals[int(len(data) * (i + 1) / max_intervals) - 1],
        ]
        for i in range(max_intervals)
    ]  # Initialize the intervals for each attribute
    return intervals


def fill_nan_mean(df, columns):
    df_ = df.copy()
    for column in columns:
        df_[column] = df_[column].fillna(df_[column].mean())
    return df_
