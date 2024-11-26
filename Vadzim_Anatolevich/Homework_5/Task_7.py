# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:53:54 2024

@author: vdmsacu
"""

import string;

class Alphabet:
    def __init__(self, lang, letters):
        self.lang = lang;
        self.letters = letters;
    def print(self):
        print(f"{self.lang} alphabet letters:", ", ".join(self.letters));
    def letters_num(self):
        print("Letters number in alphabet:", len(self.letters));
        return len(self.letters);   
    
class EngAlphabet(Alphabet):
    def __init__(self):
        super().__init__("ENG", list(string.ascii_uppercase));  
    __letters_num = len(string.ascii_uppercase);   
    def is_en_letter(self, letter):
        if letter.upper() in self.letters:
            print(f"Letter {letter.upper()} in english alphabet");
        if letter.upper() not in self.letters:
            print(f"Letter {letter.upper()} not in english alphabet");
    def letters_num(self):
        print("Letters number in alphabet:", self.__letters_num);
        return self.__letters_num;   
    @staticmethod
    def example():
        example_text = "Example text";
        print(f'"{example_text}"');
        return example_text;

EA = EngAlphabet()
EA.print();
EA.letters_num();
EA.is_en_letter("F"); 
EA.is_en_letter("Щ");   
EA.example();