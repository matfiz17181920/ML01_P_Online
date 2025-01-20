import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# Загрузка данных из CSV файла с указанными параметрами
df = pd.read_csv('C:/Users/1neon/Desktop/homework/13/credit_train.csv', sep=';', decimal=',', encoding='windows-1251')

# Обработка строковых значений в столбцах
# Удаление строк с некорректными данными
df = df.dropna(subset=['age', 'credit_month'])

# Преобразование строковых значений в числовые
df['age'] = df['age'].apply(lambda x: float(x.split('.')[0]) if isinstance(x, str) else x)
df['credit_month'] = df['credit_month'].apply(lambda x: float(x.split('.')[0]) if isinstance(x, str) else x)

# Разделение на признаки (X) и метки (y)
X = df.drop('open_account_flg', axis=1)
y = df['open_account_flg']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Определение категориальных и числовых признаков
categorical_features = ['gender', 'marital_status', 'job_position', 'education', 'living_region']
numeric_features = X.columns.difference(categorical_features)

# Создание трансформера для предобработки данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Определение моделей
gnb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())])
lr = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
lr_poly = Pipeline(steps=[('preprocessor', preprocessor), ('poly', PolynomialFeatures()), ('classifier', LogisticRegression(max_iter=1000))])
rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])

# Параметры для GridSearchCV
param_grid_lr = {
    'classifier__C': [0.1, 1, 10]
}
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200]
}
param_grid_gnb = {}

# Модели и параметры для GridSearchCV
model_params = [
    (gnb, param_grid_gnb, "GaussianNB"),
    (lr, param_grid_lr, "LogisticRegression"),
    (lr_poly, param_grid_lr, "PolynomialFeatures + LogisticRegression"),
    (rf, param_grid_rf, "RandomForestClassifier")
]

# Обучение моделей и поиск лучших параметров
for model, params, name in model_params:
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train, y_train)
    print(f"Лучшие параметры для {name}: {grid.best_params_}")
    
    # Предсказание и вычисление метрик
    y_pred = grid.predict(X_test)
    print(f"Модель: {name}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"AUC: {roc_auc_score(y_test, y_pred)}")
    print()
