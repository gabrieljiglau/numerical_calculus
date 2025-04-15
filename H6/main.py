import math
import numpy as np
from utils import f_poly, approximate_with_horner, plot_function

pi = math.pi

trig1_domain = [0, 31*pi/16]
trig2_domain = trig1_domain
trig3_domain = [0, 0, 63*pi/32]


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


def trigonometric_interpolation():
    pass


if __name__ == '__main__':
    
    poly_coefficients = poly_interpolation()
    # plot_function(1, 5, "polinomului gasit de mine", poly_coefficients[::-1])
