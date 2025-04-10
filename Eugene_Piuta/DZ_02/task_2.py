import math

print(f"Решаем уравнение типа a*x**2+b*x+c=0.")
try:
    
    a = int(input("Введите a: "))
    print(f"a: {a}.")
    b = int(input("Введите b: "))
    print(f"b: {b}.")
    c = int(input("Введите c: "))
    print(f"c: {c}.")
    
    
except Exception:
    print("Ошибка ввода данных!")
    

D = b**2 - 4*a*c
print(f"D={D}")

if D < 0:
     print("Действительных корней нет!")    
elif D==0:
     x = -b/2*a   
     print(f"Корень уравнения: x={x}.")
else:
    sqrt=math.sqrt(D)
    x1 = (-b + sqrt)/(2*a)
    x2 = (-b - sqrt)/(2*a)
    print(f"Корни уравнения: x1={x1}, x2={x2}.")