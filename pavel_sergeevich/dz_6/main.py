import string


def print_sep_section(txt: str):
    """
    Функция для красивости в консоли
    :param txt:
    :return:
    """
    len_line = 25
    print("\n" + "-" * len_line + " " + txt + " " + "-" * len_line + "\n")


class Alphabet:
    def __init__(self, lang: str, letters: list):
        # эксперименты с проверкой входных аргументов
        assert lang != "", 'Аргумент "lang" пустой'
        assert isinstance(lang, str), 'Тип аргумента "lang" не str'

        assert letters != [], 'Аргумент "letters" пустой'
        assert isinstance(letters, list), 'Тип аргумента "lang" не list'

        self.lang = lang
        self.letters = letters

    def print(self):
        for lt in self.letters:
            print(lt, end=' ')
        print('')

    def letters_num(self):
        return len(self.letters)


class EngAlphabet(Alphabet):
    __letters_num = 26

    def __init__(self):
        # инициализируем родительский класс
        super().__init__("en", list(string.ascii_lowercase))
        # а можно и так
        # Alphabet.__init__(self, "en", list(string.ascii_lowercase))

        print("Объект класса EngAlphabet создан!")

    def is_en_letter(self, char: str):
        assert len(char) == 1, "Длина полученного аргумента больше 1"

        return True if char.lower() in self.letters else False

    def letters_num(self):
        return EngAlphabet.__letters_num

    @staticmethod
    def example():
        return "lorem ipsum "*4


if __name__ == "__main__":
    print_sep_section("тест 1")
    alph_en = EngAlphabet()

    print_sep_section("тест 2")
    print("Буквы английского алфавита:")
    alph_en.print()

    print_sep_section("тест 3")
    print("Количество букв в английском алфавите равно " + str(alph_en.letters_num()))

    print_sep_section("тест 4")
    test_lt = "F"
    if alph_en.is_en_letter(test_lt):
        print("Буква \"" + test_lt + "\" есть в алфавите")
    else:
        print("Буквы \"" + test_lt + "\" нет в алфавите")

    print_sep_section("тест 5")
    test_lt = "Щ"
    if alph_en.is_en_letter(test_lt):
        print("Буква \"" + test_lt + "\" есть в алфавите")
    else:
        print("Буквы \"" + test_lt + "\" нет в алфавите")

    print_sep_section("тест 6")
    print(alph_en.example())

    # делали такое на уроке
    print_sep_section("тест *")
    print("Пример доступа к статическому свойству. Должно вывести количество букв в алфавите (26): ", end="")
    print(alph_en._EngAlphabet__letters_num)

    print(2**999)

"""
Out:

------------------------- тест 1 -------------------------

Объект класса EngAlphabet создан!

------------------------- тест 2 -------------------------

Буквы английского алфавита:
a b c d e f g h i j k l m n o p q r s t u v w x y z 

------------------------- тест 3 -------------------------

Количество букв в английском алфавите равно 26

------------------------- тест 4 -------------------------

Буква "F" есть в алфавите

------------------------- тест 5 -------------------------

Буквы "Щ" нет в алфавите

------------------------- тест 6 -------------------------

lorem ipsum lorem ipsum lorem ipsum lorem ipsum 

------------------------- тест * -------------------------

Пример доступа к статическому свойсту. Должно вывести
количество букв в алфавите (26):
26

"""