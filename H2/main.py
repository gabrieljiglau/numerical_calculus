import math

import numpy as np

from utils import compute_vector_norm, get_solution_lib, compute_norm, generate_random_array, generate_random_matrix, \
    compute_determinant

"""
gepeto pe baza: https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
"""


def luDecompositionCombined(arr_a, dU, epsilon):

    n_dims = len(arr_a)
    eps = 10 ** (-epsilon)

    combined_matrix = [[0.0 for _ in range(n_dims)] for _ in range(n_dims)]
    dL = [0.0 for _ in range(n_dims)]

    # compute row 0: L[0][0] = arr_a[0][0] / dU[0]; restul sunt 0
    if math.fabs(dU[0]) > eps:
        dL[0] = arr_a[0][0] / dU[0]
    else:
        raise ZeroDivisionError("nu se poate face impartirea la dU[0]")
    combined_matrix[0][0] = dL[0]

    # compute U[0][j] = arr_a[0][j] / L[0][0]
    for col_idx in range(1, n_dims):
        if math.fabs(dL[0]) > eps:
            combined_matrix[0][col_idx] = arr_a[0][col_idx] / dL[0]
        else:
            raise ZeroDivisionError("nu se poate face impartirea la L[0][0] in calculul U[0][%d]" % col_idx)

    # Process rows 1 through n_dims-1:
    for row_idx in range(1, n_dims):
        # așa e formula pentru primul rând, restul elementelor de pe randul lui U sunt 0
        # aici de fapt aflu elementele de pe primul rand al lui L
        if math.fabs(dU[0]) > eps:
            combined_matrix[row_idx][0] = arr_a[row_idx][0] / dU[0]
        else:
            raise ZeroDivisionError("nu se poate face impartirea la dU[0] pentru randul %d" % row_idx)

        # compute L[row_idx][col_idx]; toate elementele de sub diagonala principala din L
        for col_idx in range(1, row_idx):
            val = 0.0
            for k in range(col_idx):
                val += combined_matrix[row_idx][k] * combined_matrix[k][col_idx]
            if math.fabs(dU[col_idx]) > eps:
                combined_matrix[row_idx][col_idx] = (arr_a[row_idx][col_idx] - val) / dU[col_idx]
            else:
                raise ZeroDivisionError("nu se poate face impartirea la dU[%d]" % col_idx)

        # diagonala lui L -> L[row_idx][row_idx] (diagonal element of L) from: diagonal_math.png
        val = 0.0
        for k in range(row_idx):
            val += combined_matrix[row_idx][k] * combined_matrix[k][row_idx]
        if math.fabs(dU[row_idx]) > eps:
            dL[row_idx] = (arr_a[row_idx][row_idx] - val) / dU[row_idx]
        else:
            raise ZeroDivisionError("nu se poate face impartirea la dU[%d]" % row_idx)

        # force U'val diagonal to be dU[row_idx]
        combined_matrix[row_idx][row_idx] = dU[row_idx]

        # partea superioara a matricii: U[row_idx][j] for j = row_idx+1 to n_dims-1
        for col_idx in range(row_idx + 1, n_dims):
            val = 0.0
            for k in range(row_idx):
                val += combined_matrix[row_idx][k] * combined_matrix[k][col_idx]
            if math.fabs(dL[row_idx]) > eps:
                combined_matrix[row_idx][col_idx] = (arr_a[row_idx][col_idx] - val) / dL[row_idx]
            else:
                raise ZeroDivisionError("nu se poate face impartirea la L[%d][%d]" % (row_idx, row_idx))

    return combined_matrix, dL

def get_solution(combined_matrix, dL, target, diagonal, precision=10):

    n_dims = len(combined_matrix)
    eps = 10 ** (-precision)

    determinant = compute_determinant(dL, diagonal)
    if determinant == 0:
        print('Impossible to get a solution')
        return

    # forward substitution: L y = target
    y = [0.0] * n_dims
    for i in range(n_dims):
        sum_val = 0.0
        # L[i][j] for j < i is stored in combined_matrix[i][j]
        for j in range(i):
            sum_val += combined_matrix[i][j] * y[j]

        # If dL[i] is 1, no need to divide by it
        if math.fabs(dL[i]) < eps:
            raise ZeroDivisionError(f"Division by zero in forward substitution at index {i}")

        if dL[i] == 1.0:
            y[i] = target[i] - sum_val
        else:
            y[i] = (target[i] - sum_val) / dL[i]

    # backward substitution: U x = y
    x = [0.0] * n_dims
    for i in range(n_dims - 1, -1, -1):
        sum_val = 0.0
        # U[i][j] for j > i is stored in combined_matrix[i][j]
        for j in range(i + 1, n_dims):
            sum_val += combined_matrix[i][j] * x[j]

        # Check that U[i][i] (diagonal of U, stored in combined_matrix[i][i]) is not near zero
        if math.fabs(combined_matrix[i][i]) < eps:
            raise ZeroDivisionError(f"Division by zero in backward substitution at index {i}")

        x[i] = (y[i] - sum_val) / combined_matrix[i][i]

    return np.array(x)


if __name__ == '__main__':

    precision = 10

    forced_diagonal = [1, 1, 1]

    B = np.array([
        [2.5, 2, 2],
        [-5, -2, -3],
        [5, 6, 6.5]
    ])

    
    LU, diagL = luDecompositionCombined(B, forced_diagonal, precision)

    # For display, extract separate L and U matrices from the combined LU.
    n = len(B)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]
            elif i == j:
                L[i][j] = diagL[i]  # L's diagonal is stored separately
                U[i][j] = forced_diagonal[i]  # U's diagonal is as predefined
            else:
                U[i][j] = LU[i][j]

    print("Combined LU matrix (L below diagonal, U above and on diagonal):")
    for row in LU:
        print(row)

    print("\nL's diagonal (diagL):", diagL)

    print("\nExtracted L:")
    for row in L:
        print(row)

    print("\nExtracted U:")
    for row in U:
        print(row)

    target = np.array([2, -6, 2])

    solution = get_solution(LU, diagL, target, forced_diagonal, precision)
    print(f"my solution = {solution} has norm = {compute_norm(B, solution, target)}")

    x_lib = get_solution_lib(B, target)
    print(f"using numpy the solution is {x_lib}")

    # norm ||x_LU - x_lib||_2
    norm_1 = compute_vector_norm(solution, x_lib)

    # norm 2: ||x_LU - A_inv b||_2
    A_inv = np.linalg.inv(B)
    x_lib_b = np.dot(A_inv, target)
    norm_2 = compute_vector_norm(solution, x_lib_b)

    print(f"Euclidean norm (||x_LU - x_lib||): {norm_1}")
    print(f"Euclidean norm (||x_LU - A_inv*b||): {norm_2}")

    """
    arr = generate_random_matrix(n_dims=101)
    forced_diagonal = generate_random_array(n_dims=101)
    LU, diagL = luDecompositionCombined(arr, forced_diagonal, precision)

    # For display, extract separate L and U matrices from the combined LU.
    n = len(arr)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]
            elif i == j:
                L[i][j] = diagL[i]  # L's diagonal is stored separately
                U[i][j] = forced_diagonal[i]  # U's diagonal is as predefined
            else:
                U[i][j] = LU[i][j]

    target = generate_random_array(n_dims=101)

    solution = get_solution(LU, diagL, target, forced_diagonal, precision)
    print(f"my solution = {solution} has norm = {compute_norm(arr, solution, target)}")

    x_lib = get_solution_lib(arr, target)
    print(f"using numpy the solution is {x_lib}")

    # norm ||x_LU - x_lib||_2
    norm_1 = compute_vector_norm(solution, x_lib)

    # norm 2: ||x_LU - A_inv b||_2
    A_inv = np.linalg.inv(arr)
    x_lib_b = np.dot(A_inv, target)
    norm_2 = compute_vector_norm(solution, x_lib_b)

    print(f"Euclidean norm (||x_LU - x_lib||): {norm_1}")
    print(f"Euclidean norm (||x_LU - A_inv*b||): {norm_2}")
    """
