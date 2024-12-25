# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Шаг 1: Загрузка данных
# Загрузка данных из файла credit_train.csv с указанной кодировкой и разделителем
df = pd.read_csv('credit_train.csv', sep=';', decimal=',', encoding='windows-1251')

# Выводим количество строк в начальной таблице и первые несколько строк данных
print(f'Rows in the initial table: {len(df)}')
print(df.head())

# Шаг 2: Обработка пропусков (удаление строк с пропусками)
# Удаляем строки, содержащие пропуски данных (NaN)
df = df.dropna()

# Фильтрация числовых столбцов
# Извлекаем только числовые столбцы для дальнейшего анализа
numeric_df = df.select_dtypes(include=[np.number])

# Шаг 3: Оценка выбросов
# Вычисляем Z-оценки для числовых значений
z_scores = np.abs(stats.zscore(numeric_df))

# Удаляем строки с выбросами (где Z-оценка больше 3)
numeric_df_no_outliers = numeric_df[(z_scores < 3).all(axis=1)]

# Шаг 4: Корреляция
# Вычисляем корреляционную матрицу для числовых данных без выбросов
correlation_matrix = numeric_df_no_outliers.corr()

# Визуализация корреляционной матрицы с помощью тепловой карты (heatmap)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.show()

# Шаг 5: Графики зависимостей
# Строим графики зависимостей между числовыми переменными
sample_size = min(1000, len(numeric_df_no_outliers))
sampled_data = numeric_df_no_outliers.sample(n=sample_size, random_state=1)
sns.pairplot(sampled_data)
plt.show()

# Шаг 6: Тест на нормальность распределения
# Импортируем тест Шапиро-Уилка для проверки нормальности распределения
from scipy.stats import shapiro

# Выполняем тест Шапиро-Уилка для каждого числового столбца
for column in numeric_df_no_outliers.columns:
    stat, p = shapiro(numeric_df_no_outliers[column])
    print(f'{column}: p-value={p}')
    if p > 0.05:
        print(f'Колонка {column} имеет нормальное распределение\n')
    else:
        print(f'Колонка {column} не имеет нормального распределения\n')

# Шаг 7: Масштабирование данных
# Масштабируем данные с использованием Min-Max нормализации
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_df_no_outliers)
df_scaled = pd.DataFrame(scaled_data, columns=numeric_df_no_outliers.columns)

# Сохранение преобразованных данных в новый CSV-файл
df_scaled.to_csv('credit_train_processed.csv', index=False)
