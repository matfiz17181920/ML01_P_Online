import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
df = pd.read_csv('credit_train.csv', sep=';', decimal=',', encoding='windows-1251')

# 1. Обработка пропусков (удаление строк с пропусками)
df = df.dropna()

# Фильтрация числовых столбцов
numeric_df = df.select_dtypes(include=[np.number])

# 2. Оценка выбросов
z_scores = np.abs(stats.zscore(numeric_df))
numeric_df_no_outliers = numeric_df[(z_scores < 3).all(axis=1)]

# 3. Корреляция
correlation_matrix = numeric_df_no_outliers.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.show()

# 4. Тест на нормальность распределения
from scipy.stats import shapiro

for column in numeric_df_no_outliers.columns:
    stat, p = shapiro(numeric_df_no_outliers[column])
    print(f'{column}: p-value={p}')
    if p > 0.05:
        print(f'Колонка {column} имеет нормальное распределение\n')
    else:
        print(f'Колонка {column} не имеет нормального распределения\n')

# 5. Масштабирование данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_df_no_outliers)
df_scaled = pd.DataFrame(scaled_data, columns=numeric_df_no_outliers.columns)

# Сохранение преобразованных данных
df_scaled.to_csv('credit_train_processed.csv', index=False)
