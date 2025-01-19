"""
Задачи:
1.Загрузить файл длиной не менее 2000 символов. 
2.Составить программу, которая считает число уникальных слов в тексте (без критерия схожести)
3.Составить программу, которая считает число гласных и согласных букв. 
4.Составить программу, которая считает число предложений, их длину и число (количество) раз использования каждого слова в тексте (с критерием схожести, критерий схожести слов выбрать самостоятельно, например, spacy (en_core_web_sm) или расстояние Левенштейна). 
5.Вывести 10 наиболее часто встречаемых слов. 

План работы:
1 Загрузить файл, прочитать его и сохранить текст в переменную text.

2.1 Удалить из текста знаки пунктуации и привести текст к одному регистру.
2.2 Разбить текст на слова,
2.3 Подсчитать количество уникальных слов.

3 Подсчитать количество гласных и согласных букв.

4.1 Используя nltk разбить текст на предложения и узнать длину полученного списка.
4.2 Найти длину предложений
4.3 Найдем число (количество) раз использования каждого слова в тексте
4.3.1 Удалить стоп-слова
4.3.2 Лемматизировать полученные слова
4.3.3 Подсчить количество вхождений каждого слова

5 Отсортировать словарь и вывести на экран 10 наиболее часто встречаемых слов

"""
from string import punctuation
import nltk, copy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pymorphy2

#1 Загрузить файл, прочитать его и сохранить текст в переменную text.
with open("Eugene_Piuta/DZ_03/file.txt", encoding='utf-8') as f:
    text = f.read()


#2.1 Удаление знаков пунктуации и цифр. Приводим текст в нижний регистр.
numbers = set(range(0, 10))
exclude = set(set(punctuation)|numbers)
# Добавим дополнительные знаки в исключения
exclude.update("…", "”", "”", "“", "—", ",")
text_new =copy.copy(text)
text_new = ''.join(ch for ch in text_new if ch not in exclude)
text_new = text_new.lower()

#2.2 Разбиваем текст на слова с помощью nltk
words_nlkt = word_tokenize(text_new, language="russian")
print(f"Количество слов: {len(words_nlkt)}")

#2.3 Подсчитать количество уникальных слов.
def find_words():    
    unique_words = set()
    for word in words_nlkt:
        unique_words.add(word)
    n = len(unique_words)
    return(n)

print(f"Количество уникальных слов: {find_words()}")

# 3 Подсчитать количество гласных и согласных букв.
gls = " аяуюоеёэиы"
sogl = "бвгдйжзклмнпрстфхцчшщ"
gls_count = 0
sogl_count =0
for letter in gls:
     gls_count += text_new.count(letter)    
for letter in sogl:
     sogl_count += text_new.count(letter)    
print(f"Количество гласных букв в тексте: {gls_count}\n"
        f"Количество согласных букв в тексте: {sogl_count}")

# 4.1 Находим количество предложений с помощью nltk
#nltk.download("stopwords")     #Необходимо подгрузить один раз
#nltk.download('punkt_tab')     #Необходимо подгрузить один раз
sentences = sent_tokenize(text, language="russian")
print((f"Количество предложений в тексте: {len(sentences)}"))

# 4.2 Найдем длину предложений
sentences_len = []
for set in sentences:
    sentences_len.append(len(set))
print(f"Длины предложений: {sentences_len}")

# 4.3 Найдем число (количество) раз использования каждого слова в тексте

# 4.3.1 Удалим стоп-слова
stop_words = stopwords.words("russian")
words = [i for i in words_nlkt if i not in stop_words]

# 4.3.2 Лемматизируем полученные слова с помощью pymorphy2
morph = pymorphy2.MorphAnalyzer()
words_morph = [morph.parse(i)[0].normal_form for i in words]
print(f"{words_morph}")
#4.3.3 Подсчитаем количество вхождений каждого слова
amount = [words_morph.count(i) for i in words_morph]

# Создадим словарь для удобства и последующей сортировки
words_dict = dict(zip(words_morph, amount))

#5 Отсортируем словарь и выведем 10 наиболее часто встречаемых слов
sorted_dict = sorted(words_dict.items(), key=lambda item: item[1], reverse=True)
print(f"{sorted_dict[:10]}")

"""
Выводы:

1. Количество согласных и гласных букв в тексте примерно одинаково.
2. Уникальных слов в тексте меньше примерно в 2 раза.
3. Благодаря лемматизации мы нашли наиболее встречаемые слова в тексте.

"""