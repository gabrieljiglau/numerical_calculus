import math
import numpy as np
from utils import f_poly

pi = math.pi

trig1_domain = [0, 31*pi/16]
trig2_domain = trig1_domain
trig3_domain = [0, 0, 63*pi/32]

# np.linalg.solve() 

def poly_interpolation(x_new = 2.33, max_degree=5):

    if x_new < 0 or x_new > 5:
        print('Input x out of bounds')
        return

    num_points = max_degree
    sampled_x = np.random.random_sample(size=num_points + 1) * 5
    sampled_x = sorted(sampled_x, key=lambda x: x)
    sampled_x[0] = 0
    print(f"sampled_x = {sampled_x}")

    eval_y = []
    eval_y.append(f_poly(sampled_x[0]))  ## am adaugat 0 si f(0) artificial
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
    print(f"a_coefficients = {a_coefficients}")

    ## i) calculul lui P(x_new) cu schema lui Horner;
    ## ii) afisarea lui P(x_new)
    ## iii) suma diferentelor pt |P_m(x_i) - y_i| i= 0,..,n 


if __name__ == '__main__':
    poly_interpolation()
