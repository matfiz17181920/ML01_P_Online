<<<<<<< HEAD
#задаем функцию
def function(x):
    return -26 * x ** 2 + 25 * x - 9
#задаем производную
def derivative(x):
    return -52 * x + 25

# находим максимальное и минимальное значения 
def find_extremes(start, end, step): # это ф-я, которая принимает 3 параметра: start (-5), end (5) и step (0,1)
    min_value = float('inf') # т.о. задается "+ бесконечность", это нужно, чтобы любое значение функции позже стало меньше этого 
    max_value = float('-inf') # т.о. задается "- бесконечность", это нужно, чтобы любое значение функции позже стало больше этого
=======
def function(x):
    return -26 * x ** 2 + 25 * x - 9

def derivative(x):
    return -52 * x + 25

def find_extremes(start, end, step):
    min_value = float('inf')
    max_value = float('-inf')
>>>>>>> 282f9c2e7ad98124abd2f8e8f617d06ab1caaeef
    min_x = start
    max_x = start

    x = start
<<<<<<< HEAD
    while x <= end: #использует цикл while для перебора значений x от start до end с заданным шагом
        y = function(x) #Вызываем функцию function с текущим значением x и сохраняем результат в переменной y
        
        # проверяем наименьшее значение
        if y < min_value:
            min_value = y # если условие истинно, обновляем min_value на текущее значение y.
            min_x = x # cохраняем текущее значение x в min_x, так как оно соответствует новому наименьшему значению
        
        # проверяем наибольшее значение
=======
    while x <= end:
        y = function(x)
        
        # Проверяем наименьшее значение
        if y < min_value:
            min_value = y
            min_x = x
        
        # Проверяем наибольшее значение
>>>>>>> 282f9c2e7ad98124abd2f8e8f617d06ab1caaeef
        if y > max_value:
            max_value = y
            max_x = x

<<<<<<< HEAD
        x += step # увеличиваем значение x на заданный шаг 

    return (min_x, min_value), (max_x, max_value)

def main(): # определяем основную функцию main, где будет происходить выполнение программы
=======
        x += step

    return (min_x, min_value), (max_x, max_value)

def main():
>>>>>>> 282f9c2e7ad98124abd2f8e8f617d06ab1caaeef
    start = -5
    end = 5
    step = 0.1

    min_point, max_point = find_extremes(start, end, step)

    print(f"Наименьшее значение функции: y = {min_point[1]:.2f} при x = {min_point[0]:.2f}")
    print(f"Наибольшее значение функции: y = {max_point[1]:.2f} при x = {max_point[0]:.2f}")

    # Находим особую точку (где производная равна нулю)
    special_x = 25 / 52  # Значение x, при котором производная равна нулю
    if start <= special_x <= end:
        special_y = function(special_x)
        print(f"Особая точка (максимум): x = {special_x:.2f}, y = {special_y:.2f}")
    else:
        print("Особая точка находится вне заданного отрезка.")

if __name__ == "__main__":
    main()