import numpy as np

def compute_norm(A_init, x_LU, b):

    Ax_LU = np.dot(A_init, x_LU)
    difference = Ax_LU - b
    norm = np.linalg.norm(difference)

    return norm


def compute_vector_norm(v1, v2):
    return np.linalg.norm(v1 - v2)


def get_solution_lib(original_arr, b):
    A_inv = np.linalg.inv(original_arr)
    x_lib = np.dot(A_inv, b)
    return x_lib

def compute_determinant(dL, diagonal):
    det_L = np.prod(dL)
    det_U = np.prod(diagonal)
    det_A = det_L * det_U
    return det_A

def generate_random_matrix(n_dims, min_val=0, max_val=10):
    return np.random.uniform(min_val, max_val, size=(n_dims, n_dims))

def generate_random_array(n_dims, min_val=0, max_val=10):
    return np.random.uniform(min_val, max_val, size=n_dims)