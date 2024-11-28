# -*- coding: utf-8 -*-

import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from scipy.stats import anderson, probplot;
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler;

table = pd.read_csv('TEST.csv', delimiter = ',', dtype = str, low_memory = False);
	                
print(table);
print("Количество строк в таблице:", table.shape[0]); 

filtrated_table = table.dropna();

print("Количество строк в таблице после очистки строк с пропущенными данными:", filtrated_table.shape[0]);
deleted_rows = table.shape[0] - filtrated_table.shape[0];
print("Количество удаленных строк:", deleted_rows);

filtrated_table.loc[:, 'age'] = filtrated_table['age'].astype(int);
logical_filtrated_table_18 = filtrated_table[filtrated_table['age'] < 18];
print("Количество строк с возрастом клиента менее 18 лет:", logical_filtrated_table_18.shape[0]);
logical_filtrated_table_m_63 = filtrated_table[(filtrated_table['age'] > 63) & (filtrated_table['gender'] == 'MALE')];
print("Количество строк с возрастом клиента мужского более 63 лет:", logical_filtrated_table_m_63.shape[0]);
logical_filtrated_table_f_58 = filtrated_table[(filtrated_table['age'] > 58) & (filtrated_table['gender'] == 'FEMALE')];
print("Количество строк с возрастом клиента женского пола более 58 лет:", logical_filtrated_table_f_58.shape[0]);
print("Общее количество строк с логическими ошибками:", logical_filtrated_table_18.shape[0] + logical_filtrated_table_m_63.shape[0] + logical_filtrated_table_f_58.shape[0]);

logical_filtrated_table = filtrated_table.drop(logical_filtrated_table_18.index);
logical_filtrated_table = filtrated_table.drop(logical_filtrated_table_m_63.index);
logical_filtrated_table = filtrated_table.drop(logical_filtrated_table_f_58.index);
print("Количество строк в таблице после фильтрации логических ошибок:", logical_filtrated_table.shape[0]);

logical_filtrated_table.loc[:, 'credit_sum'] = logical_filtrated_table['credit_sum'].astype(float);
logical_filtrated_table['z_cs'] = (logical_filtrated_table['credit_sum'] - logical_filtrated_table['credit_sum'].mean()) / logical_filtrated_table['credit_sum'].std();
filtrated_table_z_cs_deleted = logical_filtrated_table[(logical_filtrated_table['z_cs'].abs() >= 3)];

filtrated_table_z = logical_filtrated_table.drop(filtrated_table_z_cs_deleted.index);
print("Количество удаленных строк в таблице после z-оценки столбца 'credit_sum':", filtrated_table_z_cs_deleted.shape[0]);
print("Количество строк в таблице после z-оценки столбца 'credit_sum':", filtrated_table_z.shape[0]);

filtrated_table_z.loc[:, 'monthly_income'] = filtrated_table_z['monthly_income'].astype(float);
filtrated_table_z['z_mi'] = (filtrated_table_z['monthly_income'] - filtrated_table_z['monthly_income'].mean()) / filtrated_table_z['monthly_income'].std();
filtrated_table_z_mi_deleted = filtrated_table_z[filtrated_table_z['z_mi'].abs() >= 3];

filtrated_table_z = filtrated_table_z.drop(filtrated_table_z_mi_deleted.index);
print("Количество удаленных строк в таблице после z-оценки столбцов 'credit_sum' и 'monthly_income':", filtrated_table_z_mi_deleted.shape[0]);
print("Количество строк в таблице после z-оценки столбцов 'credit_sum' и 'monthly_income':", filtrated_table_z.shape[0]);

filtrated_table_z.loc[:, 'credit_count'] = filtrated_table_z['credit_count'].astype(int);
filtrated_table_z['z_cc'] = (filtrated_table_z['credit_count'] - filtrated_table_z['credit_count'].mean()) / filtrated_table_z['credit_count'].std();
filtrated_table_z_cc_deleted = filtrated_table_z[filtrated_table_z['z_cc'].abs() >= 3];

filtrated_table_z = filtrated_table_z.drop(filtrated_table_z_cc_deleted.index);
print("Количество удаленных строк в таблице после z-оценки столбцов 'credit_sum', 'monthly_income' и 'credit_count':", filtrated_table_z_cc_deleted.shape[0]);
print("Количество строк в таблице после z-оценки столбцов 'credit_sum', 'monthly_income' и 'credit_count':", filtrated_table_z.shape[0]);
  
z_cs = filtrated_table_z.pop('z_cs');
z_mi = filtrated_table_z.pop('z_mi');
z_cc = filtrated_table_z.pop('z_cc');

filtrated_table_z['gender'] = filtrated_table_z['gender'].map({'MALE': 0, 'FEMALE': 1});
filtrated_table_z['marital_status'] = filtrated_table_z['marital_status'].map({'UNM': 2, 'MAR': 3, 'DIV': 4});
filtrated_table_z['job_position'] = filtrated_table_z['job_position'].map({'DIR': 5, 'SPC': 6, 'UMN': 7, 'PNA': 8, 'WOI': 9, 'BIS': 10, 'NOR': 11, 'ATP': 12, 'WRK': 13, 'INC': 14, 'INP': 15});  
filtrated_table_z['education'] = filtrated_table_z['education'].map({'GRD': 16, 'SCH': 17, 'UGR': 18, 'ACD': 19});
filtrated_table_z['living_region'] = filtrated_table_z['living_region'].map({'КРАСНОДАРСКИЙ КРАЙ': 20, 'МОСКВА': 21, 'ОБЛАСТЬ САРАТОВСКАЯ': 22, 'ОБЛАСТЬ ВОЛГОГРАДСКАЯ': 23, 'ЧЕЛЯБИНСКАЯ ОБЛАСТЬ': 24, 'СТАВРОПОЛЬСКИЙ КРАЙ': 25, 'КРАЙ СТАВРОПОЛЬСКИЙ': 25, 'ОБЛАСТЬ НИЖЕГОРОДСКАЯ': 26, 'МОСКОВСКАЯ ОБЛАСТЬ': 27, 'ОБЛАСТЬ МОСКОВСКАЯ': 27, 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУГ - ЮГРА': 28,  'САНКТ-ПЕТЕРБУРГ': 29, 'РЕСПУБЛИКА БАШКОРТОСТАН': 30,  'ОБЛАСТЬ АРХАНГЕЛЬСКАЯ': 31, 'ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНЫЙ ОКРУГ': 32, 'ПЕРМСКИЙ КРАЙ': 33, 'ПРИМОРСКИЙ КРАЙ': 34, 'РЕСПУБЛИКА КАРАЧАЕВО-ЧЕРКЕССКАЯ': 35, 'САРАТОВСКАЯ ОБЛАСТЬ': 36, 'ОБЛАСТЬ КАЛУЖСКАЯ': 37, 'ОБЛАСТЬ ВОЛОГОДСКАЯ': 38, 'РОСТОВСКАЯ ОБЛАСТЬ': 39, 'УДМУРТСКАЯ РЕСПУБЛИКА': 40, 'ОБЛАСТЬ ИРКУТСКАЯ': 41, 'ИРКУТСКАЯ ОБЛАСТЬ':41, 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ': 42, 'РЕСПУБЛИКА КОМИ': 43, 'ОБЛАСТЬ ТЮМЕНСКАЯ': 44, 'ОБЛАСТЬ БЕЛГОРОДСКАЯ': 45, 'ОБЛАСТЬ КОСТРОМСКАЯ': 46, 'РЕСПУБЛИКА ХАКАСИЯ': 47, 'РЕСПУБЛИКА ТАТАРСТАН': 48, 'ОБЛАСТЬ СВЕРДЛОВСКАЯ': 49, 'ОБЛАСТЬ ПСКОВСКАЯ': 50,  'КРАЙ ЗАБАЙКАЛЬСКИЙ': 51, 'СВЕРДЛОВСКАЯ ОБЛАСТЬ': 52, 'ОБЛАСТЬ ОРЕНБУРГСКАЯ': 53, 'ОРЕНБУРГСКАЯ ОБЛАСТЬ': 53, 'ТУЛЬСКАЯ ОБЛАСТЬ': 54, 'ОБЛАСТЬ АСТРАХАНСКАЯ': 55, 'ТАТАРСТАН РЕСПУБЛИКА': 56, 'УЛЬЯНОВСКАЯ ОБЛАСТЬ': 57, 'ОБЛАСТЬ АМУРСКАЯ':58, 'ОБЛАСТЬ САМАРСКАЯ': 59,  'ОБЛАСТЬ ВЛАДИМИРСКАЯ': 60, 'РЕСПУБЛИКА ЧЕЧЕНСКАЯ': 61, 'РЕСПУБЛИКА АДЫГЕЯ': 62});   

filtrated_table_z = filtrated_table_z.dropna();
pearson_correlation = filtrated_table_z.corr(method = 'pearson');
print(pearson_correlation);

plt.figure(figsize = (8, 6));
sns.heatmap(pearson_correlation, annot = True, cmap = 'coolwarm', fmt = ".2f", square = True);
plt.title('Тепловая карта корреляции по критерию Пирсона');
plt.show();



filtrated_table_z['credit_sum'] = pd.to_numeric(filtrated_table_z['credit_sum'], errors='coerce');

result_cs = anderson(filtrated_table_z['credit_sum'], dist = 'norm');
print("Статистика теста для столбца 'credit_sum':", result_cs.statistic);
print("Критические значения для столбца 'credit_sum':", result_cs.critical_values);
print("Уровни значимости для столбца 'credit_sum':", result_cs.significance_level);
if result_cs.statistic < result_cs.critical_values[2]:  # Использование критического значения для 5% уровня значимости
    print("Распределение для столбца 'credit_sum' является нормальным");
else:
    print("Распределение для столбца 'credit_sum' не является нормальным");

plt.figure(figsize = (15, 6));
plt.subplot(1, 3, 1);
sns.histplot(filtrated_table_z['credit_sum'], kde = True);
plt.title("Гистограмма и график плотности для столбца 'credit_sum'");

plt.subplot(1, 3, 2);
probplot(filtrated_table_z['credit_sum'], dist = "norm", plot = plt);
plt.title("Q-Q график для столбца 'credit_sum'");

plt.subplot(1, 3, 3);
sns.boxplot(x = filtrated_table_z['credit_sum']);
plt.title("БД с ограничителями выбросов для столбца 'credit_sum'");

plt.tight_layout();
plt.show();



filtrated_table_z['monthly_income'] = pd.to_numeric(filtrated_table_z['monthly_income'], errors='coerce');

result_mi = anderson(filtrated_table_z['monthly_income'], dist = 'norm');
print("Статистика теста для столбца 'monthly_income':", result_mi.statistic);
print("Критические значения для столбца 'monthly_income':", result_mi.critical_values);
print("Уровни значимости для столбца 'monthly_income':", result_mi.significance_level);
if result_mi.statistic < result_mi.critical_values[2]:  
    print("Распределение для столбца 'monthly_income' является нормальным");
else:
    print("Распределение для столбца 'monthly_income' не является нормальным");

plt.figure(figsize = (15, 6));
plt.subplot(1, 3, 1);
sns.histplot(filtrated_table_z['monthly_income'], kde = True);
plt.title("Гистограмма и график плотности для столбца 'monthly_income'");


plt.subplot(1, 3, 2);
probplot(filtrated_table_z['monthly_income'], dist = "norm", plot = plt);
plt.title("Q-Q график для столбца 'monthly_income'");

plt.subplot(1, 3, 3);
sns.boxplot(x = filtrated_table_z['monthly_income']);
plt.title("БД с ограничителями выбросов для столбца 'monthly_income'");

plt.tight_layout();
plt.show();



filtrated_table_z['score_shk'] = pd.to_numeric(filtrated_table_z['score_shk'], errors='coerce');

result_ss = anderson(filtrated_table_z['score_shk'], dist = 'norm');
print("Статистика теста для столбца 'score_shk':", result_ss.statistic);
print("Критические значения для столбца 'score_shk':", result_ss.critical_values);
print("Уровни значимости для столбца 'score_shk':", result_ss.significance_level);
if result_ss.statistic < result_ss.critical_values[2]:  
    print("Распределение для столбца 'score_shk' является нормальным");
else:
    print("Распределение для столбца 'score_shk' не является нормальным");

plt.figure(figsize = (15, 6));
plt.subplot(1, 3, 1);
sns.histplot(filtrated_table_z['score_shk'], kde = True);
plt.title("Гистограмма и график плотности для столбца 'score_shk'");


plt.subplot(1, 3, 2);
probplot(filtrated_table_z['score_shk'], dist = "norm", plot = plt);
plt.title("Q-Q график для столбца 'score_shk'");

plt.subplot(1, 3, 3);
sns.boxplot(x = filtrated_table_z['score_shk']);
plt.title("БД с ограничителями выбросов для столбца 'score_shk'");

plt.tight_layout();
plt.show();



filtrated_table_z['tariff_id'] = pd.to_numeric(filtrated_table_z['tariff_id'], errors='coerce');

result_ti = anderson(filtrated_table_z['tariff_id'], dist = 'norm');
print("Статистика теста для столбца 'tariff_id':", result_ti.statistic);
print("Критические значения для столбца 'tariff_id':", result_ti.critical_values);
print("Уровни значимости для столбца 'tariff_id':", result_ti.significance_level);
if result_ti.statistic < result_ti.critical_values[2]:  
    print("Распределение для столбца 'tariff_id' является нормальным");
else:
    print("Распределение для столбца 'tariff_id' не является нормальным");

plt.figure(figsize = (15, 6));
plt.subplot(1, 3, 1);
sns.histplot(filtrated_table_z['tariff_id'], kde = True);
plt.title("Гистограмма и график плотности для столбца 'tariff_id'");


plt.subplot(1, 3, 2);
probplot(filtrated_table_z['score_shk'], dist = "norm", plot = plt);
plt.title("Q-Q график для столбца 'tariff_id'");

plt.subplot(1, 3, 3);
sns.boxplot(x = filtrated_table_z['tariff_id']);
plt.title("БД с ограничителями выбросов для столбца 'tariff_id'");

plt.tight_layout();
plt.show();



filtrated_table_z['credit_count'] = pd.to_numeric(filtrated_table_z['credit_count'], errors='coerce');

result_cc = anderson(filtrated_table_z['credit_count'], dist = 'norm');
print("Статистика теста для столбца 'credit_count':", result_cc.statistic);
print("Критические значения для столбца 'credit_count':", result_cc.critical_values);
print("Уровни значимости для столбца 'credit_count':", result_cc.significance_level);
if result_cc.statistic < result_cc.critical_values[2]:  
    print("Распределение для столбца 'credit_count' является нормальным");
else:
    print("Распределение для столбца 'credit_count' не является нормальным");

plt.figure(figsize = (15, 6));
plt.subplot(1, 3, 1);
sns.histplot(filtrated_table_z['credit_count'], kde = True);
plt.title("Гистограмма и график плотности для столбца 'credit_count'");


plt.subplot(1, 3, 2);
probplot(filtrated_table_z['credit_count'], dist = "norm", plot = plt);
plt.title("Q-Q график для столбца 'credit_count'");

plt.subplot(1, 3, 3);
sns.boxplot(x = filtrated_table_z['credit_count']);
plt.title("БД с ограничителями выбросов для столбца 'credit_count'");

plt.tight_layout();
plt.show();



min_max_scaler = MinMaxScaler();
filtrated_table_z['credit_sum_min_max'] = min_max_scaler.fit_transform(filtrated_table_z[['credit_sum']]);
filtrated_table_z['monthly_income_min_max'] = min_max_scaler.fit_transform(filtrated_table_z[['monthly_income']]);
filtrated_table_z['score_shk_min_max'] = min_max_scaler.fit_transform(filtrated_table_z[['score_shk']]);
filtrated_table_z['tariff_id_min_max'] = min_max_scaler.fit_transform(filtrated_table_z[['tariff_id']]);
filtrated_table_z['credit_count_min_max'] = min_max_scaler.fit_transform(filtrated_table_z[['credit_count']]);


plt.subplot(1, 2, 1)
sns.histplot(filtrated_table_z['credit_sum_min_max'], kde = True, color = 'blue', label = 'credit_sum', alpha = 0.6);
sns.histplot(filtrated_table_z['monthly_income_min_max'], kde = True, color = 'orange', label = 'monthly', alpha = 0.6);
sns.histplot(filtrated_table_z['score_shk_min_max'], kde = True, color = 'red', label = 'score_shk', alpha = 0.6);
sns.histplot(filtrated_table_z['tariff_id_min_max'], kde = True, color = 'black', label = 'tariff_id', alpha = 0.6);
sns.histplot(filtrated_table_z['credit_count_min_max'], kde = True, color = 'beige', label = 'credit_count', alpha = 0.6);

plt.title('Гистограмма нормализованных значений');
plt.xlabel('Нормализованные значения');
plt.ylabel('Частота');
plt.legend();
plt.show();