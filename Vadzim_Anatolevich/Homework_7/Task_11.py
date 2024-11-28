# -*- coding: utf-8 -*-

p_a = 0.01;
p_b_a = 0.98;
p_b_n_a = 0.02;  
p_n_a = 1 - p_a;

p_b = p_b_a * p_a + p_b_n_a * p_n_a;
p_a_g_b = (p_b_a * p_a) / p_b;

print(f'Вероятность: {p_a_g_b * 100:.2f}%');