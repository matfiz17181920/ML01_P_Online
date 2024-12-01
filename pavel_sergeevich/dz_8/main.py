import numpy
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
import scipy

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sns.set_theme()


def sns_plot_example():
    sns.set_theme()
    data = sns.load_dataset("iris")

    # draw lineplot
    sns.lineplot(x="sepal_length", y="sepal_width", data=data)
    plt.show()


def get_adv_info(ds: pd.DataFrame):
    msg = "-" * 120 + "\n"

    msg += "Количество фич в датасете: " + str(ds.shape[1]) + "\n"
    msg += "Количество строк в датасете: " + str(ds.shape[0]) + "\n"

    msg += "\n"
    msg += "Названия фич:\n"
    for cl in ds.columns:
        msg += "    - " + cl + "\n"
    msg += "\n"

    msg += "Используемая память (MB): " + str(round(ds.memory_usage().sum()/1024/1024, 3)) + "\n"

    msg += "-" * 120 + "\n"

    print(msg)


def get_estimate_losses(ds: pd.DataFrame, flag_quiet: bool) -> pd.DataFrame:
    def get_bar_of_null(name: str):
        pass_lines_u = ds[ds[name].isnull()]["living_region"].value_counts()

        pass_lines_df = pd.DataFrame(pass_lines_u.to_frame())
        pass_lines_df = pass_lines_df.reset_index()
        pass_lines_df.columns = ['unique', 'counts']
        pass_lines_df = pass_lines_df[pass_lines_df["counts"] > 100]

        f, ax = plt.subplots()
        sns.barplot(x="counts", y="unique", data=pass_lines_df, label="count").set_title(name)
        ax.legend(ncol=2, loc="lower right", frameon=True)
        sns.despine(left=True, bottom=True)

    print("Справка по пропускам в датасете:")
    print(ds.isnull().sum())

    # попытка определить зависимость пропусков данных от региона
    if not flag_quiet:
        get_bar_of_null("overdue_credit_count")
        get_bar_of_null("credit_count")

    # удаляем пропуски
    original_num_lines = ds.shape[0]
    ds = ds.dropna()
    update_num_lines = ds.shape[0]

    print("Число удаленных строк - " + str(original_num_lines - update_num_lines))

    est_losses = round(100 - update_num_lines/original_num_lines*100, 2)

    print()
    print("Всего потери данных на пропусках составляют: " + str(est_losses) + "%\n")
    print("-" * 120)

    return ds


def analyze_outlier(ds: pd.DataFrame, flag_quiet=False):
    def plot_outliers(ds: pd.DataFrame):
        fig, axs = plt.subplots(ncols=3, nrows=2)
        sns.boxplot(x='age', data=ds, ax=axs[0][0])
        sns.boxplot(x='credit_sum', data=ds, ax=axs[0][1])
        sns.boxplot(x='credit_month', data=ds, ax=axs[0][2])
        sns.boxplot(x='score_shk', data=ds, ax=axs[1][0])
        sns.boxplot(x='monthly_income', data=ds, ax=axs[1][1])
        sns.boxplot(x='credit_count', data=ds, ax=axs[1][2])

        fig, axs = plt.subplots(ncols=1)
        sns.boxplot(x='tariff_id', data=ds, ax=axs)

        fig, axs = plt.subplots(ncols=2, nrows=2)
        sns.histplot(x='open_account_flg', data=ds, ax=axs[0][0])
        sns.boxplot(x='open_account_flg', data=ds, ax=axs[1][0])
        sns.histplot(x='overdue_credit_count', data=ds, ax=axs[0][1])
        sns.boxplot(x='overdue_credit_count', data=ds, ax=axs[1][1])

    def clear_outliers(ds: pd.DataFrame):
        column_list = ["age", "credit_sum", "credit_month", "monthly_income", "credit_count"]

        index_list = []
        for cm_name in column_list:
            q1 = ds[cm_name].quantile(0.25)
            q3 = ds[cm_name].quantile(0.75)
            iqr = q3 - q1
            outliers = ds[(ds[cm_name] < (q1 - 1.5 * iqr)) | (ds[cm_name] > (q3 + 1.5 * iqr))]

            index_list = index_list + outliers.index.to_list()

        print("Количество индексов на удаление со всех столбцов: " + str(len(index_list)))
        print("Количество уникальных индексов на удаление: " + str(len(list(set(index_list)))))
        ds = ds.drop(index=list(set(index_list)))

        return ds

    sh_start = ds.shape[0]

    if not flag_quiet:
        plot_outliers(ds)
    ds = clear_outliers(ds)
    if not flag_quiet:
        plot_outliers(ds)

    sh_finish = ds.shape[0]
    sh_dif = sh_start - sh_finish

    print("Количество выбросов: " + str(sh_dif) + " (" + str(round(sh_dif/sh_start*100, 2)) + "%)")
    print("Данных в наборе осталось: " + str(sh_finish))

    outlier_sum = ds[(ds["open_account_flg"] == 1.0)].shape[0]
    print("Выбросы значений в фиче open_account_flg: ", outlier_sum)
    print("Отношение к общему количеству: " + str(round(outlier_sum/ds.shape[0]*100, 2)) + "%")

    outlier_0 = ds[(ds["overdue_credit_count"] == 1.0)].shape[0]
    outlier_1 = ds[(ds["overdue_credit_count"] == 2.0)].shape[0]
    outlier_2 = ds[(ds["overdue_credit_count"] == 3.0)].shape[0]
    outlier_sum = outlier_0 + outlier_1 + outlier_2
    print("Выбросы значений в фиче overdue_credit_count: ", outlier_0, outlier_1, outlier_2)
    print("Отношение к общему количеству: " + str(round(outlier_sum/ds.shape[0]*100, 2)) + "%")

    return ds


def cat_data_processing(ds: pd.DataFrame, column_list: list):
    for cm_name in column_list:
        encoder = LabelEncoder()
        ds[cm_name] = encoder.fit_transform(ds[cm_name])

    return ds


def scaler_processing(ds: pd.DataFrame):
    scaler = MinMaxScaler()

    # формируем список названий колонок игнорируя client_id, т.е. это не существенная информация для модели
    col = ds.columns.to_list()
    col.remove('client_id')

    ds[col] = scaler.fit_transform(ds[col])

    return ds


def get_norm_estimate(ds: pd.DataFrame, column_list: list):
    for cm_name in column_list:
        stat, p = stats.normaltest(ds[cm_name])
        # print('Statistics=%.3f, p-value=%.3f' % (stat, p))

        alpha = 0.05
        if p > alpha:
            print('Принять гипотезу о нормальности колонки ' + cm_name)
        else:
            print('Отклонить гипотезу о нормальности колонки ' + cm_name)


def get_corr(ds: pd.DataFrame):
    ds = ds.drop(columns=['client_id'])
    ds_corr = ds.corr()

    sns.heatmap(ds_corr, annot=True, fmt=".2f", annot_kws={"size": 8})


if __name__ == "__main__":
    path2data = "data/data_set.csv"

    data_set = pd.read_csv(path2data, encoding='WINDOWS-1251', sep=";", decimal=',')
    # print(data_set["credit_sum"].describe())

    # функция вывода мета-данных датасета
    get_adv_info(data_set)

    # поиск, оценка и удаление пропусков
    data_set = get_estimate_losses(data_set, flag_quiet=True)

    # анализ выбросов
    data_set = analyze_outlier(data_set, flag_quiet=True)

    # обработка категориальных данных
    cat = ['gender', 'marital_status', 'job_position', 'education', 'living_region']
    data_set = cat_data_processing(data_set, cat)

    # print(data_set["living_region"].describe())

    # масштабируем данные
    data_set = scaler_processing(data_set)
    print("-" * 120)

    # получаем оценку нормальности данных
    norm_estimate_list = ['age', 'credit_sum', 'credit_month', 'score_shk', 'monthly_income', 'credit_count', 'tariff_id']
    get_norm_estimate(data_set, norm_estimate_list)

    get_corr(data_set)

    plt.show()

