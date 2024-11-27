def cache_three_runs(func):
    cache = {}
    call_count = 0

    def wrapper(*args, **kwargs):
        nonlocal call_count

        if call_count < 3 and args in cache:
            print("Возвращаем кэшированный результат")
            call_count += 1
            return cache[args]
        else:
            print("Выполняем функцию и кэшируем результат")
            result = func(*args, **kwargs)
            cache[args] = result
            call_count = 1  # Сбрасываем счетчик вызовов после обновления кэша
            return result

    return wrapper

@cache_three_runs
def square(x):
    return x * x

# Тестирование
print(square(4))  # Выполняет функцию и кэширует результат
print(square(4))  # Возвращает кэшированный результат
print(square(4))  # Возвращает кэшированный результат
print(square(4))  # Возвращает кэшированный результат
print(square(4))  # Выполняет функцию и кэширует результат снова
