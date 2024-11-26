import time

def cache(func):
    def wrapper(*args, **kwargs):
        read_cache = open('cache.txt', "r").read()
        write_cache = open('cache.txt', "w")
        data = read_cache.split()
        if int(data[0]) != 3 :
            print("+1 запуск")
            write_cache.write(f"{int(data[0]) + 1} {int(data[1])}")
            return data[1]
        else:
            print("Перезапись")
            write_cache.write(f"{0} {func(*args, **kwargs)}")
            return func(*args, **kwargs)
    return wrapper

@cache
def get_current_time(time_plus):
    return time.localtime().tm_sec * time_plus

print(get_current_time(2))