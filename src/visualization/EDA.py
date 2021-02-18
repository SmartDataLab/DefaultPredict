import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def draw_density_plot(df, columns, save_path, xlim=None):
    figure = plt.figure()
    df[columns].plot(kind="density")
    if xlim:
        plt.xlim(xlim)
    plt.savefig(save_path)
    plt.close(figure)


def draw_accumulation_plot(df, save_path):
    df = df.copy()
    df["count"] = 1
    df["APPLYDATE"] = df["APPLYDATE"].apply(
        lambda x: pd.to_datetime(x.replace("/", "-"))
    )
    df = df.sort_values("APPLYDATE")
    series = df.groupby("APPLYDATE")["count"].sum()
    figure = plt.figure()
    series.cumsum().plot()
    plt.ylabel("累积贷款人数")
    plt.savefig(save_path)
    plt.close(figure)


def draw_boxplot(df_2019, df_2020, save_path):
    figure = plt.figure()
    df_2019_ = df_2019.copy()
    df_2019_["year"] = 2019
    df_2020_ = df_2020.copy()
    df_2020_["year"] = 2020
    df_plot = pd.concat([df_2019_, df_2020_])
    for i, column in enumerate(list(df_2019.columns)[:20]):
        plt.subplot(5, 4, 1 + i)
        sns.boxplot(y=column, x="year", data=df_plot)
        plt.ylabel("")
    plt.savefig(save_path)
    plt.close(figure)


def draw_corr_heat_map(cor_feature, save_path):
    figure = plt.figure()
    cor_df = cor_feature.corr()
    cor_mat = np.array(cor_df)
    sns.heatmap(cor_mat, square=True, cbar=True)
    plt.savefig(save_path, dpi=300)
    plt.close(figure)


def draw_pandemic_contrast(model1, model2, data1, data2, save_path):
    figure = plt.figure()
    plt.subplot(312)
    result_proba = model1.predict_proba(data2)[:, 1]
    result_proba.sort()
    plt.plot(range(len(result_proba)), result_proba)
    plt.xlabel("2020")
    plt.subplot(313)
    # beta_x = np.log(proba / (1 - proba))
    # mu = mu_fun(beta_x, -3.23)
    # mu.sort()
    result_proba = model2.predict_proba(data2)[:, 1]
    result_proba.sort()
    # plt.plot(range(len(result_proba2)), mu)
    plt.plot(range(len(result_proba)), result_proba)
    plt.xlabel("2020(auto-adjusted)")
    plt.subplot(311)
    result_proba = model1.predict_proba(data1)[:, 1]
    result_proba.sort()
    plt.plot(range(len(result_proba)), result_proba)
    plt.xlabel("2019")
    plt.savefig(save_path, dpi=500)  # 'figure/all_year_proba_curve.png'
    plt.close(figure)