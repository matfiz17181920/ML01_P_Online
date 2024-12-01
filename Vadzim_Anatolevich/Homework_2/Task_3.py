#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Nov  3 18:58:39 2024

@author: vdzmshacu
"""

import array as arr
from sympy import diff, symbols, solve

#Определение функции
def equation_func (p_1, p_2, stp):
    
    """
    Функция находит значения функции на заданном интервале с заданным шагом, ее производную и точку экстремума функции
    Возвращает массив значений "y" функции от х, максимальное и минимальное значения функции, точку экстремума функции
    """
    
    y = arr.array('f', []); 
    x = float(p_1);
    print(f"Вычисление значений для функции вида y=-26*x(^2)+25*x-9 \n x:\t\t y:");
    while x <= p_2:
        t = -26 * x ** (2) + 25 * x - 9;
        y.append(t);
        print(f"{x:.2f}     {y[-1]:.2f}");
        x += stp; 
    max_y = max(y);
    min_y = min(y);
    print(f"Максимальное значение функции в интервале от {p_1:.2f} до {p_2:.2f}: {max_y:.2f}");
    print(f"Минимальное значение функции в интервале от {p_1:.2f} до {p_2:.2f}: {min_y:.2f}");
    x = symbols('x', real = True);
    diff_equation = diff(-26 * x ** (2) + 25 * x - 9, x);
    print(f"Частная производная первого порядка функции вида y=-26*x(^2)+25*x-9 равна: (d/dx)(-26*x(^2)+25*x-9) = {diff_equation}");
    critical_point = solve(diff_equation, x);
    print(f"Критическая точка функции: x = {critical_point}");
    return y, max_y, min_y, critical_point;  
 
#Программа для вычисления наибольшего и наименьшего значений функции
    
#Начальное определение шага
step = 0;

#Конструкция try-except для проверки првильности ввода
while step <= 0:
    try:        
        step = float(input("Введите шаг вычисления функции: "));
        if step <= 0:
            print("Шаг вычисления функции не может быть отрицательной величиной или равнятся нулю!");
    except:
        continue;
        
#Определение интервала определения функции
point_1 = -5;
point_2 = 5;

#Вызов функции
equation_func(point_1, point_2, step);

