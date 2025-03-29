import copy
import numpy as np
from utils import *

def find_inverse(matrix, identity_matrix, method, num_iterations=10000, precision=6):

    previous_vk = np.array(build_first_matrix(matrix, precision))
    current_vk = previous_vk.copy()

    n_dims = len(matrix)
    matrix = np.array(matrix)

    epsilon = 10 ** (-precision)
    delta_v = 10 **3

    k = 1
    while k < num_iterations + 1 and delta_v >= epsilon and delta_v <= 10 ** 10:
        
        print(f"k = {k}")

        previous_vk = current_vk.copy()
        current_vk = build_next_matrix(matrix, identity_matrix, current_vk, method)
        delta_v = np.linalg.norm(current_vk - previous_vk)

        k += 1

    if delta_v < epsilon:
        print(f"Found vk = {current_vk}")
        resulted_norm = np.dot(matrix, current_vk)
        resulted_norm -= identity_matrix
        print(f"norm: ||A * found_inverse - I|| = {np.linalg.norm(resulted_norm)}")

        true_inv = inverse_matrix(len(matrix))
        print(f"true_inv = {true_inv}")

        print(f"norm||true_inv - found_inv|| = {np.linalg.norm(current_vk - true_inv)}")
    else:
        print("Divergence !!")


if __name__ == '__main__':
    
    I = np.array(build_identity_matrix(10))
    A = np.array(build_input_matrix(10))
    method = "schultz" # schultz / li1 / li2
    find_inverse(A, I, method, num_iterations=10000)

    # print(inverse_matrix(5))
    # print(np.linalg.inv(A))