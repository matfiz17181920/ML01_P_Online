import math
from sympy import diff, symbols

print(f"Решаем уравнение типа -26*x**2+25*x-9.")
a = -26
b = 25
c = -9  

print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
x = -5
R = -26*x**2+25*x-9
max = R
min = R
while x < 5:
    x = round((x + 0.1),2)
  #  print(f"x = {x}")
    R = -26*x**2+25*x-9
    if R>max:
        max = R
     #   print(f"Обнаружен новый максимум: {max}")
    elif R < min:
        min = R
     #   print(f"Обнаружен новый минимум: {min}")
    continue
print(f"Максимум функции: {max}")
print(f"Минимум функции: {min}")
    
# Находим производную функции
x = symbols('x')

derivative = diff(-26*x**2+25*x-9, x)
print(f"Производная: {derivative}")

# Находим особую точку функции : derivative=0
# -52x+25=0
# x=25/52
singular_point = 25/52
 

print(f"Особая точка: {singular_point}")
