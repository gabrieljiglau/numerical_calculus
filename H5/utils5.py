import random
import numpy as np
import pandas as pd


def is_matrix_symmetric(rows, epsilon):

    for row_idx in range(len(rows)):
        for col_idx, value in rows[row_idx]:
            
            found = False

            if not any(row == row_idx and abs(val - value) < epsilon for row, val in rows[col_idx]):
                return False
    
    return True

def build_symmetric_matrix(n_dims, sparsity=0.1, max_value=10):

    matrix = [[] for _ in range(n_dims)]
    for row_idx in range(n_dims):
        for col_idx in range(row_idx):  ## upper triangular only
            if random.random() < sparsity:
                value = random.randint(1, max_value)
                matrix[row_idx].append((col_idx, value))
                if row_idx != col_idx:  ## symmetric AROUND the diagonal
                    matrix[col_idx].append((row_idx, value))

    return matrix

def generate_unit_vector(n_dims):

    vector = np.random.rand(n_dims)
    norm = np.linalg.norm(vector)  ## ord=2 by default
    return vector / norm

def get_matrix_product(rows, v0, n_dims):

    results = np.zeros(len(v0))
    for i in range(n_dims):
        current_result = 0
        for col_idx, value in rows[i]:
            current_result += np.dot(v0[i], value)
        
        results[i] = current_result
    return results

if __name__ == '__main__':
    
    n_dims = 99
    # print(build_symmetric_matrix(n_dims))
    v = generate_unit_vector(n_dims)
    print(np.linalg.norm(v, ord=2))
