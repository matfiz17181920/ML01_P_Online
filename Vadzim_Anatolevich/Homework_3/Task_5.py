#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:57:32 2024

@author: vdzmshacu
"""

import glob;
import spacy;
from collections import Counter;

nlp = spacy.load("ru_core_news_sm");
path = 'Chehov_Anton__Tolstyi_i_tonkii.txt';

for file in glob.glob(path):
    with open('Chehov_Anton__Tolstyi_i_tonkii.txt', encoding='utf-8', errors='ignore') as file_in:
        text = file_in.read();
        text_analysis = text.split();
        text_analysis = nlp(text);
        words = [token.text for token in text_analysis if not token.is_stop and not token.is_punct];
        word_freq = Counter(words);
        common_words = word_freq.most_common(10);
        print(f"10 Наиболее часто встречаемых слов и символов:", common_words);
        unique_words = [word for (word, freq) in word_freq.items() if freq == 1];
        print(f"Уникальных слов (с учетом критерия схожести)", len(unique_words));