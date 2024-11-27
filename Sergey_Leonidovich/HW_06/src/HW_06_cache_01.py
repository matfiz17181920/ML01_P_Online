"""
ЗАДАНИЕ.
Напишите декоратор, оптимизирующий работу декорируемой функции.
Декоратор должен сохранять результат работы функции на ближайшие три запуска и вместо выполнения функции возвращать сохранённый результат.
После трёх запусков функция должна вызываться вновь, а результат работы функции — вновь кешироваться.
"""

from functools import wraps


def cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        cache_key = (func, args, tuple(kwargs.items()))

        print(f"Function called {wrapper.cnt} times")

        if wrapper.cnt < 4:
            wrapper.cnt += 1

            if cache_key in wrapper.cache:
                print("Cached data...")
                return wrapper.cache[cache_key]
            else:
                wrapper.cache[cache_key] = func(*args, **kwargs)
                return wrapper.cache[cache_key]
        else:
            wrapper.cnt = 1
            wrapper.cache[cache_key] = func(*args, **kwargs)
            return wrapper.cache[cache_key]

    wrapper.cache = {}
    wrapper.cnt = 0

    return wrapper


@cache
def expensive_function(a, b):
    print(f"Calculating data...")
    return a + b


if __name__ == "__main__":

    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")  # clear cache
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(3, 3)}\n")  # other *args
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")
    print(f"Result: {expensive_function(2, 3)}\n")  # clear cache


