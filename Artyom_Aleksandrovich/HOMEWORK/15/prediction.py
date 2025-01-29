# Импортируем необходимые библиотеки
import pandas as pd

# Загрузка данных
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')

# Просмотр первых строк данных
print(train_data.head())
print(test_data.head())
# Объединяем тренировочные и тестовые данные для предобработки
data = pd.concat([train_data, test_data], sort=False)

# Заполнение пропущенных значений
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Создание новых признаков
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 1  # Одинокий = 1, иначе = 0
data['IsAlone'].loc[data['FamilySize'] > 1] = 0

# Преобразование категориальных признаков
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Разделение данных обратно на тренировочные и тестовые
train_data = data[:len(train_data)]
test_data = data[len(train_data):]

# Просмотр первых строк данных после предобработки
print(train_data.head())
print(test_data.head())
import seaborn as sns
import matplotlib.pyplot as plt

# Анализ зависимости выживаемости от размера семьи
sns.barplot(x='FamilySize', y='Survived', data=train_data)
plt.title('Выживаемость в зависимости от размера семьи')
plt.show()

# Анализ зависимости выживаемости от пола
sns.barplot(x='Sex_male', y='Survived', data=train_data)
plt.title('Выживаемость в зависимости от пола')
plt.show()

# Анализ зависимости выживаемости от возраста
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack')
plt.title('Выживаемость в зависимости от возраста')
plt.show()

# Анализ зависимости выживаемости от класса каюты
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Выживаемость в зависимости от класса каюты')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Подготовка данных для обучения и тестирования
X = train_data.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Логистическая регрессия
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print('Logistic Regression metrics:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred_logreg):.2f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_logreg):.2f}')

# Дерево решений
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print('Decision Tree metrics:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred_tree):.2f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_tree):.2f}')

# Случайный лес
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print('Random Forest metrics:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_forest):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred_forest):.2f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_forest):.2f}')
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()

