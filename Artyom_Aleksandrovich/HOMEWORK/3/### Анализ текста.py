### Анализ текста
# Импортируем все необходимое
    ## import re: Импортирует модуль для работы с регулярными выражениями, который позволяет выполнять сложные текстовые манипуляции.
    ## from collections import Counter: Импортирует класс Counter из модуля collections, который используется для подсчета объектов (например, слов) в списке.
    ## import spacy: Импортирует библиотеку Spacy, которая предназначена для обработки естественного языка.
    ## from Levenshtein import ratio: Импортирует функцию ratio из библиотеки Levenshtein для вычисления расстояния Левенштейна, которое измеряет схожесть между двумя строками.
    ## from spacy.lang.en.stop_words import STOP_WORDS: Импортирует предопределенный список стоп-слов из Spacy, которые часто не несут значимого смысла в тексте (например, "и", "но", "или").
    ## from string import punctuation: Импортирует строку, содержащую все знаки препинания (например, ".", ",", "!").
import re
from collections import Counter
import spacy
from Levenshtein import ratio
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

    ## Функция read_file принимает путь к файлу (file_path) и открывает файл в режиме чтения с кодировкой UTF-8.
    ## Читает содержимое файла и возвращает его как строку.
# Функция для чтения файла
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

    ## Функция clean_text очищает текст, приводя его к нижнему регистру.
    ## Удаляет все знаки препинания с помощью регулярного выражения.
    ## Разбивает текст на отдельные слова.
    ## Исключает стоп-слова из списка слов.
    ## Возвращает очищенный текст в виде строки.
# Функция для очистки текста от стоп-слов и знаков препинания
def clean_text(text):
    text = text.lower()  # Приведение текста к нижнему регистру
    text = re.sub(f'[{re.escape(punctuation)}]', '', text)  # Удаление знаков препинания
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]  # Исключение стоп-слов
    return ' '.join(words)

    ## Функция count_unique_words разбивает текст на слова.
    ## Преобразует список слов в множество, чтобы удалить дубликаты.
    ## Возвращает количество уникальных слов.
# Функция для подсчета уникальных слов
def count_unique_words(text):
    words = text.split()
    unique_words = set(words)
    return len(unique_words)

    ## Функция count_vowels_consonants определяет списки гласных и согласных букв.
    ## Использует генераторы списка для подсчета гласных и согласных в тексте.
    ## Возвращает количество гласных и согласных букв.
# Функция для подсчета гласных и согласных
def count_vowels_consonants(text):
    vowels = "аеёиоуыэюяaeiou"
    consonants = "бвгджзйклмнпрстфхцчшщbcdfghjklmnpqrstvwxyz"
    
    vowels_count = sum(1 for char in text if char in vowels)
    consonants_count = sum(1 for char in text if char in consonants)
    
    return vowels_count, consonants_count

    ## Функция analyze_sentences_words загружает небольшую модель Spacy.
    ## Применяет модель к тексту, чтобы разметить предложения и слова.
    ## Извлекает предложения и подсчитывает их количество.
    ## Вычисляет длину каждого предложения.
    ## Извлекает слова и подсчитывает их частоту.
# Функция для подсчета предложений, их длины и частоты слов с критерием схожести
def analyze_sentences_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    
    sentence_lengths = [len(sent.text.split()) for sent in sentences]
    
    words = [token.text.lower() for token in doc if token.is_alpha]
    word_freq = Counter(words)
    
    return num_sentences, sentence_lengths, word_freq

    ## Функция process_words_with_similarity итерирует по списку слов и проверяет их схожесть с уникальными словами, используя расстояние Левенштейна.
    ## Если слово достаточно похоже на уже встреченное слово, увеличивает его счетчик частоты.
    ## Если слово уникально, добавляет его в список уникальных слов и увеличивает его счетчик.
# Функция для обработки слов с использованием расстояния Левенштейна
def process_words_with_similarity(words, threshold=0.8):
    unique_words = []
    word_freq = Counter()
    
    for word in words:
        is_unique = True
        for uw in unique_words:
            if ratio(word, uw) > threshold:
                word_freq[uw] += 1
                is_unique = False
                break
        if is_unique:
            unique_words.append(word)
            word_freq[word] += 1
    
    return word_freq

# Функция для вывода топ-10 часто встречаемых слов
def get_top_words(word_freq, top_n=10):
    return word_freq.most_common(top_n)

# Основная программа
file_path = r'C:\Users\1neon\Desktop\fightclub.txt'  # Указанный путь к файлу
text = read_file(file_path)

# Очистка текста
cleaned_text = clean_text(text)

# Подсчет уникальных слов
unique_words_count = count_unique_words(cleaned_text)
print(f"Число уникальных слов: {unique_words_count}")

# Подсчет гласных и согласных
vowels_count, consonants_count = count_vowels_consonants(cleaned_text)
print(f"Число гласных: {vowels_count}")
print(f"Число согласных: {consonants_count}")

# Анализ предложений и слов
num_sentences, sentence_lengths, word_freq = analyze_sentences_words(cleaned_text)
print(f"Число предложений: {num_sentences}")
print(f"Длины предложений: {sentence_lengths}")

# Обработка слов с критерием схожести
words = cleaned_text.split()
similarity_word_freq = process_words_with_similarity(words)

# Топ-10 часто встречаемых слов
top_words = get_top_words(similarity_word_freq)
print("\nТоп-10 часто встречаемых слов:")
for word, freq in top_words:
    print(f"'{word}': {freq} раз")
