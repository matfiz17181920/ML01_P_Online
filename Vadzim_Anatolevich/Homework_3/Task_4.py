#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:18:02 2024

@author: vdzmshacu
"""

import re;

def words_list_creator(file):
    text = file.read();
    text = text.replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").replace("-", " ");
    text = text.lower();
    words = text.split();
    words_list = list(words);
    return words_list;

def uniq_words_counter(words_list):
    uniq_words = list();
    for word in words_list:
        if word in uniq_words:
            pass;
        else:
            uniq_words.append(word);
    return len(uniq_words);

def vowels_and_consonants_counter(words_list):
    string = str(words_list);
    vowels_list = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"];
    consonants_list = ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "х", "ш", "щ", "ч"];
    vowels_count = 0;
    consonants_count = 0;
    for letter in string:
        if letter in vowels_list:
            vowels_count +=1;
        if letter in consonants_list:
            consonants_count +=1;
    return [vowels_count, consonants_count];

def sentences_counter(file):
    text = file.read();
    sentences = re.findall(r"\s*([^.?!]+)\s*", text);
    sentence_count = 0;
    while sentence_count < len(sentences):
        sentence_string = str(sentences[sentence_count]);
        sentence_string = sentence_string.replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").replace("-", " ");
        sentence_string = sentence_string.lower();
        sentence_string_words = sentence_string.split();
        sentence_string_list = list(sentence_string_words);
        print(f"Длина предложения {sentence_count + 1}:", len(sentence_string_list), "слов(-а, -о)");
        sentence_count +=1;
    return len(sentences);

file = open('Chehov_Anton__Tolstyi_i_tonkii.txt', 'r', encoding = 'utf-8');
print(f"Число уникальных слов в тесте (без критерия схожести): {uniq_words_counter(words_list_creator(file))}");
file = open('Chehov_Anton__Tolstyi_i_tonkii.txt', 'r', encoding = 'utf-8');
print (f"Число гласных букв в тексте: {vowels_and_consonants_counter(words_list_creator(file))[0]}");
file = open('Chehov_Anton__Tolstyi_i_tonkii.txt', 'r', encoding = 'utf-8');
print (f"Число согласных букв в тексте: {vowels_and_consonants_counter(words_list_creator(file))[1]}");
file = open('Chehov_Anton__Tolstyi_i_tonkii.txt', 'r', encoding = 'utf-8');
print(f"Количество предложений в тексте: {sentences_counter(file)}");

    
    
    
    
    




        
             