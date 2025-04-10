
try:
    
    S = float(input("Введите длину маршрута(км): "))
    if S<=0:
        raise Exception
    print(f"Длина маршрута: {S}км.")
    R = float(input("Введите расход на 100км(л): "))
    if R<=0:
        raise Exception
    print(f"Расход на 100км: {R}км.")
    V = (R*S)/100
    print(f"Расход топлива на {S}км. составит {V}л.")
except Exception:
    print("Ошибка ввода данных!")
    