import math
import numpy as np


def build_first_matrix(matrix, precision):
    
    epsilon = 10 ** (-precision)
    numerator = np.transpose(matrix)
    n_dims = len(matrix)
    

    col_max = np.max(np.sum(np.abs(matrix), axis=0))  # A_1

    row_max = np.max(np.sum(np.abs(matrix), axis=1))  # A_inf

    denumerator = col_max * row_max
    if math.fabs(denumerator) > epsilon:
        return numerator / (col_max * row_max)
    else:
        print('The denominator is almost 0')
  
    return -1

def is_first_matrix_acceptable(original_matrix, identity_matrix, v0):

    norm = np.linalg.norm(np.dot(original_matrix, v0) - identity_matrix)
    print(f"norm = {norm}")
    return True if norm < 1 else False

def build_empty_matrix(n_dims):

    return np.array([[0 for _ in range(n_dims)] for _ in range(n_dims)])

def build_identity_matrix(n_dims):

    arr = build_empty_matrix(n_dims)
    for i in range(n_dims):
        arr[i][i] = 1

    return arr

def build_input_matrix(n_dims):

    arr = build_empty_matrix(n_dims)
    for i in range(n_dims):
        arr[i][i] = 1
        if i + 1 < n_dims:
            arr[i + 1][i] = 2

    return arr


def identity_minus_avk(original_matrix, identity_matrix, current_vk, constant):
    
    negated_matrix = -original_matrix
    new_matrix = np.dot(negated_matrix, current_vk)

    np.fill_diagonal(new_matrix, np.diag(new_matrix) + constant)
    return new_matrix

def next_matrix_schultz(original_matrix, identity_matrix, current_vk):

    """
    the input matrices must be a numpy array
    """

    paranthesis = identity_minus_avk(original_matrix, identity_matrix, current_vk, 2)
    return np.dot(current_vk, paranthesis)

def next_matrix_li1(original_matrix, identity_matrix, current_vk):
    
    """
    the input matrices must be a numpy array
    """

    paranthesis = identity_minus_avk(original_matrix, identity_matrix, current_vk, 3)
    triple_i = 3 * identity_matrix
    a_vk = np.dot(original_matrix, current_vk)

    return np.dot(current_vk, triple_i - np.dot(a_vk, paranthesis))

def next_matrix_li2(original_matrix, identity_matrix, current_vk):

    """
    the input matrices must be a numpy array
    """

    a_vk = np.dot(original_matrix, current_vk)
    paranthesis1 = identity_minus_avk(original_matrix, identity_matrix, current_vk, 3)
    paranthesis2 = identity_minus_avk(original_matrix, identity_matrix, current_vk, 1)

    return np.dot(current_vk, identity_matrix + 0.25 * np.dot(paranthesis1, paranthesis2))


def build_next_matrix(original_matrix, identity_matrix, current_vk, method):

    if method == 'schultz':
        return next_matrix_schultz(original_matrix, identity_matrix, current_vk)
    elif method == 'li1':
        return next_matrix_li1(original_matrix, identity_matrix, current_vk)
    elif method == 'li2':
        return next_matrix_li2(original_matrix, identity_matrix, current_vk)
    else:
        print('Improper method for building the next matrix')
        return -1

def inverse_matrix(n_dims):
    I, J = np.indices((n_dims, n_dims))
    inv = np.where(I >= J, (-2.0) ** (I - J), 0)
    return inv

if __name__ == '__main__':

    A = [
        [-3, 5, 7],
        [2, 6, 4],
        [0, 2, 8]
    ]    

    identity_matrix = build_identity_matrix(3)
    print(f"identity_matrix = {identity_matrix}")

    """
    precision = 10
    V0 = get_first_matrix(A, precision)
    # print(f"V0 = {V0}")
    print(is_first_matrix_acceptable(A, identity_matrix, V0))
    """

    print(add_constant_on_diagonal(identity_matrix, 2))

