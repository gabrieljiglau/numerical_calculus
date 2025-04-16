import math
import numpy as np
from utils import f_poly, approximate_with_horner, plot_function, eval_trig_function, approximateT

pi = math.pi


def poly_interpolation(num_points=1001, x_new = 2.33, max_degree=2):

    if x_new < 1 or x_new > 5:
        print('Input x out of bounds')
        return

    """
    sampled_x = 4 * np.random.random_sample(size=num_points) + 1  ## (upper - lower) * rand(0, 1) + lower
    sampled_x = sorted(sampled_x, key=lambda x: x)
    """

    sampled_x = [1]
    sampled_x.extend(np.random.uniform(low=1, high=5, size=num_points - 2))
    sampled_x.append(5)
   
    eval_y = []
    for i in range(num_points):
        eval_y.append(f_poly(sampled_x[i]))
    
    B = np.zeros((max_degree + 1, max_degree + 1))
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            current_sum = 0
            for x in sampled_x:
                current_sum += x ** (i + j)
            B[i][j] = current_sum

    right_side = []
    for i in range(max_degree + 1):
        current_sum = 0
        for k in range(num_points):
            current_sum += eval_y[k] * sampled_x[k] ** i

        right_side.append(current_sum)

    # B * a = f
    a_coefficients = np.linalg.solve(B, right_side)
    print(f"a_coefficients = {a_coefficients}")  # highest degree to lowest 

    print(f"f(x_new) = {f_poly(x_new)}")

    eval_x_new = approximate_with_horner(a_coefficients[::-1], x_new)
    print(f"P(x_new) = {eval_x_new}")

    abs_difference = abs(eval_x_new - f_poly(x_new))
    print(f"|P(x_new) - f(x_new)| = {abs_difference}")

    least_squares_diff = 0
    for xi, yi in zip(sampled_x, eval_y):
        least_squares_diff += abs(approximate_with_horner(a_coefficients[::-1], xi) - yi)
    print(f"sum(|P(x_i) - y_i|) = {least_squares_diff}")

    return a_coefficients


def trigonometric_interpolation(maximum_frequency, trig_domain, trig_function, x_new=3/2 * pi): 


    ## maximum_frequency = maximum k used in sin and cos
    num_samples = 2 * maximum_frequency + 1
    x_samples = np.random.uniform(low=trig_domain[0], high=trig_domain[1], size=num_samples)

    T = np.zeros((num_samples, num_samples))
    for i in range(num_samples):  # row
        for j in range(num_samples):  # column
            if j == 0:
                T[i][j] = 1
            elif j % 2 == 1:
                k = (j + 1) // 2
                T[i][j] = math.sin(k * x_samples[i])
            elif j % 2 == 0:
                k = j // 2
                T[i][j] = math.cos(k * x_samples[i])
    
    eval_y = []
    for x in x_samples:
        y = eval_trig_function(trig_function, x)
        eval_y.append(y)

    # B * a = f
    x_coefficients = np.linalg.solve(T, eval_y)
    
    eval_x_new = approximateT(x_coefficients, x_new, maximum_frequency)
    print(f"f(x_new) = {eval_trig_function(trig_function, x_new)}")
    print(f"T(x_new) = {eval_x_new}")

    abs_diff = abs(eval_x_new - eval_trig_function(trig_function, x_new))
    print(f"|T(x_new) - f(x_new)| = {abs_diff}")

    return x_coefficients



if __name__ == '__main__':
    
    """
    poly_coefficients = poly_interpolation()
    plot_function(1, 5, "polinomului gasit de mine", poly_coefficients[::-1])
    """

    trig1_domain = [0, 31*pi/16]     # f1
    trig2_domain = trig1_domain      # f2
    trig3_domain = [0, 0, 63*pi/32]  # f3
    
    m = 4
    trig_coef = trigonometric_interpolation(m, trig1_domain, 'f1')
    plot_function(trig1_domain[0], trig1_domain[1], "functiei trigonometrice", trig_coef, 
                    trig_interpolation=True, m =4)
