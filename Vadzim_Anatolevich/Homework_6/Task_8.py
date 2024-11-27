# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:03:03 2024

@author: vdmsa
"""

def cache(limit=3):
    def decorator_function(func):
        cache = [];

        def cache_function(*args):
            if args in cache:
                result = cache[cache.index(args)];
                print("Значение функции есть в кэше. Возврат значения из кэша:");
                return result;
            result = func(*args);
            print(f"Вычисление функции: {result}");
            if len(cache) < limit:
                cache.append(args);
            else:
                cache.pop(0);
                cache.append(args);
            print(f"Значение функции добавлено в кэш: {result}");
            return result;
        return cache_function;
    return decorator_function;
@cache(3)
def function(x):
    return x;
print(function(2))  
print(function(2));  
print(function(3)); 
print(function(3)); 
print(function(5)); 
print(function(2));  
print(function(6));  
print(function(3));  
print(function(2));
