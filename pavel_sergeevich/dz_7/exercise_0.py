"""
Задание:
    Напишите декоратор, оптимизирующий работу декорируемой функции. Декоратор должен сохранять результат работы функции
    на ближайшие три запуска и вместо выполнения функции возвращать сохранённый результат. 
    После трёх запусков функция должна вызываться вновь, а результат работы функции — вновь кешироваться.

Комментарии:
    целью задания является отработка использования декораторов, ввиду чего кэширование функции реализовано наивно и 
    как указано в задании, без добавления излишней сложности. Что бы не плодить глобальных переменных и тем самым,
    лишних зависимостей кэш присоединяю к функции (возможно, т.к. функция в Python - объект). Для тестирования
    декоратора написал 3 функции.

Выводы:
    изучил чуть подробнее как работают декораторы - как их писать и применять
"""

import random


def cache_wrapper(func):
    def wrapped(*args, **kwargs):
        try:
            _ = func.__cache__
            _ = func.__cache_itt__
        except AttributeError:
            func.__cache__ = None
            func.__cache_itt__ = 0

        if func.__cache__ is not None and func.__cache_itt__ < 2:
            func.__cache_itt__ += 1
        else:
            print("Update context for " + func.__name__ + " (" + str(id(func)) + ")")
            func.__cache_itt__ = 0
            func.__cache__ = func(*args, **kwargs)

        return func.__cache__
    return wrapped


@cache_wrapper
def func_for_test_0(a: int):
    return 2 ** a


@cache_wrapper
def func_for_test_1(a: int):
    return a * a * a


@cache_wrapper
def func_for_test_2(a: int):
    return (358 * a) % 10


if __name__ == "__main__":
    print("-" * 120)

    # запускаем три разные функции со случайным аргументом. Если это первый или каждый третий вызов функции, то
    # она выполняется, если нет - выдается значение из кэша
    for i in range(10):
        print("Iteration " + str(i) + "\n")
        r_0 = func_for_test_0(random.randint(0, 10))
        r_1 = func_for_test_1(random.randint(0, 10))
        r_2 = func_for_test_2(random.randint(0, 10))

        print()
        print("Results of functions: func_0 = " + str(r_0) + " func_1 = " + str(r_1) + " func_2 = " + str(r_2))
        print()
        print("-"*120)


"""
------------------------------------------------------------------------------------------------------------------------
Iteration 0

Update context for func_for_test_0 (1911582561760)
Update context for func_for_test_1 (1911582562048)
Update context for func_for_test_2 (1911582562336)

Results of functions: func_0 = 2 func_1 = 729 func_2 = 2

------------------------------------------------------------------------------------------------------------------------
Iteration 1


Results of functions: func_0 = 2 func_1 = 729 func_2 = 2

------------------------------------------------------------------------------------------------------------------------
Iteration 2


Results of functions: func_0 = 2 func_1 = 729 func_2 = 2

------------------------------------------------------------------------------------------------------------------------
Iteration 3

Update context for func_for_test_0 (1911582561760)
Update context for func_for_test_1 (1911582562048)
Update context for func_for_test_2 (1911582562336)

Results of functions: func_0 = 16 func_1 = 512 func_2 = 6

------------------------------------------------------------------------------------------------------------------------
Iteration 4


Results of functions: func_0 = 16 func_1 = 512 func_2 = 6

------------------------------------------------------------------------------------------------------------------------
Iteration 5


Results of functions: func_0 = 16 func_1 = 512 func_2 = 6

------------------------------------------------------------------------------------------------------------------------
Iteration 6

Update context for func_for_test_0 (1911582561760)
Update context for func_for_test_1 (1911582562048)
Update context for func_for_test_2 (1911582562336)

Results of functions: func_0 = 32 func_1 = 729 func_2 = 8

------------------------------------------------------------------------------------------------------------------------
Iteration 7


Results of functions: func_0 = 32 func_1 = 729 func_2 = 8

------------------------------------------------------------------------------------------------------------------------
Iteration 8


Results of functions: func_0 = 32 func_1 = 729 func_2 = 8

------------------------------------------------------------------------------------------------------------------------
Iteration 9

Update context for func_for_test_0 (1911582561760)
Update context for func_for_test_1 (1911582562048)
Update context for func_for_test_2 (1911582562336)

Results of functions: func_0 = 256 func_1 = 64 func_2 = 8

------------------------------------------------------------------------------------------------------------------------
"""