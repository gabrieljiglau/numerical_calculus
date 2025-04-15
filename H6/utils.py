import math
import numpy as np
import matplotlib.pyplot as plt

def f_poly(x_in):
    return x_in**4  - 12*(x_in**3) + 30*(x_in**2) + 12

def f1_trig(x_in):
    return math.sin(x_in) - math.cos(x_in)

def f2_trig(x_in):
    return math.sin(2 * x_in) + math.sin(x_in) + math.cos(3 * x_in)

def f3_trig(x_in):
    return math.sin(x_in) ** 2 - math.cos(x_in) ** 2

def eval_trig_function(trig_function, x_in):

    if trig_function == 'f1':
        return f1_trig(x_in)
    elif trig_function == 'f2':
        return f2_trig(x_in)
    elif trig_function == 'f3':
        return f3_trig
    else:
        print(f"Unkown input function specified \n")
        print("Recognized names: f1, f2 or f3")
        return -1

def approximateT(x_coefficients, x_new, m):

    result = 1  # fi0(x) = 1
    for i in range(2 * m):
        result += x_coefficients[i] * math.sin(i * x_new) if i % 2 == 0 else x_coefficients[i] * math.cos(i * x_new)

    return result


def approximate_with_horner(a_coefficients, x_new):

    result = a_coefficients[0]
    for i in range(1, len(a_coefficients)):
        result = a_coefficients[i] + result * x_new

    return result

def plot_function(lower_bound, upper_bound, function_title, coefficients, f_name=None, size=100):

    x_points = np.random.uniform(low=lower_bound, high=upper_bound, size=size)
    if f_name:
        y_points = [f_name(x) for x in x_points]
    else:
        y_points = [approximate_with_horner(coefficients, x) for x in x_points]

    plt.plot(x_points, y_points, 'o')
    plt.xlabel('Punctul x')
    plt.ylabel('Evaluarea lui f(x)')
    plt.title(f"Graficul {function_title}")
    plt.show()


if __name__ == '__main__':

    pi = math.pi
    trig1_domain = [0, 31*pi/16]
    plot_function(f1_trig, trig1_domain[0], trig1_domain[1], "primei funcții trigonometrice")
