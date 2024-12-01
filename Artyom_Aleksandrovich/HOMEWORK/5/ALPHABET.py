
###Класс АЛФАВИТ
import string

class Alphabet:
    def __init__(self, lang, letters):
        self.lang = lang
        self.letters = letters
    
    def print(self):
        print("Алфавит:", " ".join(self.letters))
    
    def letters_num(self):
        return len(self.letters)

class EngAlphabet(Alphabet):
    __letters_num = 26

    def __init__(self):
        super().__init__('En', list(string.ascii_uppercase))
    
    def is_en_letter(self, letter):
        return letter.upper() in self.letters
    
    def letters_num(self):
        return self.__letters_num
    
    staticmethod
    def example():
        return "Happiness is only real when shared"
#ТЕСТ
#Создаем объект класса EngAlphabet
eng_alphabet = EngAlphabet()

#Напечатаем буквы алфавита для этого объекта
eng_alphabet.print()

#Выведем количество букв в алфавите
print("Количество букв в алфавите:", eng_alphabet.letters_num())

#Проверим, относится ли буква F к английскому алфавиту
print("Буква 'F' относится к английскому алфавиту:", eng_alphabet.is_en_letter('F'))

#Проверим, относится ли буква Щ к английскому алфавиту
print("Буква 'Щ' относится к английскому алфавиту:", eng_alphabet.is_en_letter('Щ'))

#Выведем пример текста на английском языке
print("Пример текста на английском языке:", EngAlphabet.example())

###Класс АЛФАВИТ
import string

class Alphabet:
    def __init__(self, lang, letters):
        self.lang = lang
        self.letters = letters
    
    def print(self):
        print("Алфавит:", " ".join(self.letters))
    
    def letters_num(self):
        return len(self.letters)

class EngAlphabet(Alphabet):
    __letters_num = 26

    def __init__(self):
        super().__init__('En', list(string.ascii_uppercase))
    
    def is_en_letter(self, letter):
        return letter.upper() in self.letters
    
    def letters_num(self):
        return self.__letters_num
    
    staticmethod
    def example():
        return "Happiness is only real when shared"
#ТЕСТ
#Создаем объект класса EngAlphabet
eng_alphabet = EngAlphabet()

#Напечатаем буквы алфавита для этого объекта
eng_alphabet.print()

#Выведем количество букв в алфавите
print("Количество букв в алфавите:", eng_alphabet.letters_num())

#Проверим, относится ли буква F к английскому алфавиту
print("Буква 'F' относится к английскому алфавиту:", eng_alphabet.is_en_letter('F'))

#Проверим, относится ли буква Щ к английскому алфавиту
print("Буква 'Щ' относится к английскому алфавиту:", eng_alphabet.is_en_letter('Щ'))

#Выведем пример текста на английском языке
print("Пример текста на английском языке:", EngAlphabet.example())
