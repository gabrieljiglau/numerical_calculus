import copy
import numpy as np
from utils import *

def find_inverse(matrix, identity_matrix, method, num_iterations=10000, precision=6):

    previous_vk = np.array(build_first_matrix(matrix, precision))
    current_vk = identity_matrix

    n_dims = len(matrix)
    matrix = np.array(matrix)

    epsilon = 10 ** (-precision)
    delta_v = 0

    k = 1
    while k < num_iterations + 1 and delta_v >= epsilon and delta_v <= 10 ** 10:

        if k == 1:
            current_vk = copy.deepcopy(previous_vk)
        else:       
            current_vk = build_next_matrix(matrix, identity_matrix, current_vk) 

        delta_v = abs(current_vk - previous_vk)

        k += 1

    if delta_v < epsilon:
        print(f"Found vk = {current_vk}")
    else:
        print("Divergence !!")


if __name__ == '__main__':
    
    I = np.array(build_empty_matrix(5))
    A = np.array(build_input_matrix(5))
    method = "schultz" # schultz / li1 / li2
    find_inverse(A, I, method)