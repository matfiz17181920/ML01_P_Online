import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import spacy
from scipy import spatial

# Чтение файла
file_name = 'fightclub.txt'
try:
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    print('Файл успешно прочитан')
except:
    print('Ошибка чтения файла!')

# Преобразование в нижний регистр
text = text.lower()

# Замена неалфавитных знаков на пробелы
text_no_signs = "".join([c if c.isalpha() else " " for c in text])
word_list = text_no_signs.split()

# Определение стоп-слов
STOP_WORDS = set("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

s same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such 

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
throughout thus to together too top toward towards twelve twenty two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split())
contractions = ["n't", "nt", "'d", "d", "'ll", "ll", "'m", "m", "'re", "re", "'s", "s", "'ve", "ve"]
STOP_WORDS.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        STOP_WORDS.add(stopword.replace("'", apostrophe))

# Фильтрация стоп-слов
word_list_new = [aword for aword in word_list if aword not in STOP_WORDS]
word_list = list(set(word_list_new))

# Токенизация и удаление стоп-слов с использованием nltk
nltk.download('punkt')
nltk.download('stopwords')
tokens = word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

# Векторизация слов с помощью Word2Vec
model = Word2Vec(sentences=[filtered_tokens], vector_size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv

# PCA для уменьшения размерности
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors.vectors)

# Построение графика уменьшенной размерности
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(word_vectors.index_to_key):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

# Построение графика доли объяснённой дисперсии
pca = PCA().fit(word_vectors.vectors)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Количество компонент')
plt.ylabel('Кумулятивная доля объяснённой дисперсии')
plt.show()

# Определение числа компонент для сохранения 90% дисперсии
explained_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(explained_variance >= 0.9) + 1
print(f"Число компонент для сохранения 90% дисперсии: {num_components}")

# Восстановление пары исходных слов
word = 'fight'  # Замените на любое слово из вашего списка
similar_words = model.wv.most_similar(word, topn=2)
print(similar_words)

# Вывод первых 10 слов из словаря модели
print(word_vectors.index_to_key[:10])

# Загрузка модели spacy
nlp = spacy.load('en_core_web_md')
doc_spacy = nlp(" ".join([aword for aword in word_list]))

# Получение векторов для отдельных токенов
for token in doc_spacy:
    print(f"Токен: {token.text}, Вектор: {token.vector}")
print(token.vector.shape)

# Векторный массив
tmp = [token.vector for token in doc_spacy if token.has_vector]
words_vectorized = [token.text for token in doc_spacy if token.has_vector]
vector_array = np.array(tmp)
print(vector_array.shape)
initial_vector_length = token.vector.shape[0]

# Масштабирование (опционально)
"""
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(vector_array)
"""
X = vector_array

# PCA для уменьшения размерности
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(f'Общая доля объяснённой дисперсии: {100 * sum(pca.explained_variance_ratio_):.2f}%')

n_sp = 0
var_ratio = []
previous_expl_disp_value = 0
current_expl_disp_value = 0
pca = PCA()
pca.fit(X)
specified_d_value = 90

# Определение числа компонент, обеспечивающих 90% объяснённой дисперсии
for i in range(initial_vector_length+1):
    current_expl_disp_value = np.sum(pca.explained_variance_ratio_[:i])
    var_ratio.append(current_expl_disp_value)
    if current_expl_disp_value >= specified_d_value / 100 and previous_expl_disp_value < specified_d_value / 100:
        n_sp = i
        print(f'Число компонент, обеспечивающих {specified_d_value}% объяснённой дисперсии: {n_sp}')
    previous_expl_disp_value = current_expl_disp_value

# Построение графика доли объяснённой дисперсии
plt.figure(figsize=(4, 2), dpi=150)
plt.grid()
plt.plot(var_ratio, marker='o')
plt.xlabel('Количество компонент')
plt.ylabel('Доля объяснённой дисперсии')
plt.title('Количество компонент против доли объяснённой дисперсии')
plt.show()

print(f'Количество использованных главных компонент: {n_sp} из {initial_vector_length}')
pca_sp = PCA(n_components=n_sp)
X_sp = pca_sp.fit_transform(X)
X_recovered = pca_sp.inverse_transform(X_sp)
print(X_recovered.shape)
print(X.shape)

# Построение KDTree для поиска векторов в начальной таблице
tree = spatial.KDTree(X)
n = 0
N = 1399
print('Пары слов до PCA и после восстановления:')
for i in range(N):
    x_recovered = X_recovered[i, :]
    position = tree.query(x_recovered)[1]
    print(f'{words_vectorized[i]} - {words_vectorized[position]}')
    if i == position:
        n += 1
    else:
        print('!!! обнаружено различие !!!')
print(f'Процент совпадений: {n / N * 100:.2f}%')
