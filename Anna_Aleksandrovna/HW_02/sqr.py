<<<<<<< HEAD
import math #задаем нужную библиотеку

def function1(a, b, c): #задаем формулу дискриминанта
    D = b ** 2 - 4 * a * c
    
    if D > 0: #нахдим корни
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        return x1, x2
    elif D == 0: #случай, если дискриминант = 0
        root = -b / (2 * a)
        return root,
    else:
        return None # None - специальный тип данных означающий "ничего"

def main(): # определяем основную функцию main, где будет происходить выполнение программы
    try: # вводим вручную данные a, b, c
=======
import math

def function1(a, b, c):
    D = b ** 2 - 4 * a * c
    
    if D > 0:
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        return x1, x2
    elif D == 0:
        root = -b / (2 * a)
        return root,
    else:
        return None

def main():
    try:
>>>>>>> 282f9c2e7ad98124abd2f8e8f617d06ab1caaeef
        a = float(input("Введите коэффициент a (не равно 0): "))
        if a == 0:
            raise ValueError("Коэффициент a не должен быть равен 0.")
        
        b = float(input("Введите коэффициент b: "))
        c = float(input("Введите коэффициент c: "))

        roots = function1(a, b, c)

        if roots is None:
            print("Уравнение не имеет действительных корней.")
        elif len(roots) == 1:
            print(f"У уравнения один двойной корень: {roots[0]:.2f}")
        else:
            print(f"У уравнения два действительных корня: {roots[0]:.2f} и {roots[1]:.2f}")

    except ValueError as e:
        print(f"Ошибка ввода: {e}")

if __name__ == "__main__":
    main()