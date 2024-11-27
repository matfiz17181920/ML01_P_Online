#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Nov  3 17:25:41 2024

@author: vdzmshacu
"""

#Программа для расчета расхода топлива

#Начальные значения длины пути и расхода топлива для конструкции try-except
path_length = 0;
fuel_consumption = 0;

#Ввод значений длины пути и расхода топлива (проверка правильности введенных величин осуществляется...
#...применением конструкции try-except)
while path_length <= 0:
    try:        
        path_length = float(input("Введите длину маршрута в км: "));
        if path_length <= 0:
            print("Длина маршрута не может быть отрицательной величиной или равнятся нулю!");
    except:
        continue;
while fuel_consumption <= 0:
     try:        
         fuel_consumption = float(input("Введите расход топлива на 100 км пути в литрах: "));
         if fuel_consumption <= 0:
             print("Величина расхода топлива не может быть отрицательной величиной или равнятся нулю!");
     except:
         continue;     

#Вычисление количества потраченого топлива
fuel_use = (path_length * fuel_consumption)/100;

#Вывод результата
print("Количество топлива, затраченного на", path_length, "км пути, при расходе", fuel_consumption, f"л/100 км, составляет: {fuel_use:.2f} л.");