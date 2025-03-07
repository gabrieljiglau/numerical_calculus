import math

import matplotlib.pyplot as plt
import numpy as np

from H1.utils import *


def ex1(power=0, base=10):
    for _ in range(200):
        if (1 + pow(base, -power)) == 1:
            return power - 1, pow(base, -(power - 1))
        # cu power găsesc numărul pentru care expresia ajunge == 1;
        # ,dar eu îl doresc pe cel mai mare care nu avem egalitate
        else:
            power += 1


def is_addition_not_associative(first_num=1.0, second_num=1e-15, third_num=1e-15):
    a = (first_num + second_num) + third_num
    b = first_num + (second_num + third_num)
    print(f"a = {a}, b = {b}")
    return (first_num + second_num) + third_num != first_num + (second_num + third_num)


"""
how to sort a dictionary based on the value
https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
"""


def approximate_sinus(trials=10000):
    computing_times = {i: 0 for i in range(8)}
    total_times = []
    losses = {}
    print(losses)

    pi = math.pi
    for i in range(trials):
        generated_num = double_between(-pi / 2, pi / 2)
        squared_num = generated_num ** 2
        true_val = math.sin(generated_num)

        polynomials, current_times = calculate_ps(generated_num, squared_num)
        total_times = [sum(times) for times in current_times]
        losses = compute_error(polynomials, true_val)

    sorted_polynomials = sorted(losses.items(), key=lambda kv: (kv[1], kv[0]))
    print(f"polynomials by accuracy:\n {sorted_polynomials}")

    for i in range(8):
        computing_times[i] = total_times[i]

    computing_times = sorted(computing_times.items(), key=lambda kv: (kv[1], kv[0]))

    print(f"computing times:\n {computing_times}")
    return sorted_polynomials, computing_times

def plot_times(computing_times):
    """
    si aici gepeto
    """
    x_vals, y_vals = zip(*computing_times)
    plt.bar(x_vals, y_vals, color='skyblue', edgecolor='black')
    plt.xlabel('Polynomial index')
    plt.ylabel('Computing times(seconds)')
    plt.title('Computation time per polynomial')
    plt.xticks(np.arange(min(x_vals), max(x_vals) + 1, 1))
    plt.show()


if __name__ == '__main__':
    # num = print(ex1())  # 1e-15

    # print(is_addition_not_associative())

    # print(get_associative_factors(100, 1000, 15.0))

    _, compute_times = approximate_sinus(1000)
    plot_times(compute_times)
