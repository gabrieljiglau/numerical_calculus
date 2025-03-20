import numpy as np
from utils import *

n_dims = [10000, 20000, 30000, 80000, 2025]  # a_plus_b are n_dims = 2025 

def gauss_seidel_solver(matrix_df, target_df, n_dims, max_iterations=10000, representation1=True, precision=15):
    
    epsilon = 10 ** (-precision)
    diagonal = get_diagonal(matrix_df, n_dims)
    
    if is_diagonal_null(diagonal):
        print("Cannot perform Gauss-Seidel if any element from the diagonal is 0")
        return

    rows = build_reprsesentation(matrix_df, n_dims, representation1)

    ## make the algorithm work with the second, dictionary based, representation
    x_gs = np.zeros((n_dims, 2))
    
    k = 0
    delta_x = 1
    
    while delta_x >= epsilon and delta_x <= 10 ** 8 and k < max_iterations:
        
        for i in range(n_dims):
            
            b_i = target_df.values[i][0] 
            
            sum_aij_xij = get_sum(rows[i], x_gs, i, representation1)
            new_val = (b_i - sum_aij_xij) / diagonal[i]  # Gauss-Seidel update.
            
            x_gs[i, 1] = new_val

           
        delta_x = np.linalg.norm(x_gs[:, 1] - x_gs[:, 0])
        x_gs[:, 0] = x_gs[:, 1]
        k += 1
        print(f"Iteration {k}: delta_x = {delta_x}")
        
        if delta_x < epsilon:
            break

    if delta_x < epsilon:
        print(f"Converged after {k} iterations. x_approx = {x_gs[:, 1]}")
        a_x = get_product(rows, diagonal, x_gs, n_dims, representation1)
        residual = np.array(get_difference(a_x, target_df))
        print(f"residual = {residual[0:20]}")

        residual = np.linalg.norm(residual, ord=np.inf)
        print(f"Residual norm = {residual}")  # A_inf
    else:
        print("Divergence!")

def add_matrices(matrix1_df, matrix2_df, target_df, n_dims, representation1, precision=10):

    epsilon = 10 ** (-precision)

    rows1 = build_reprsesentation(matrix1_df, n_dims, representation1)
    rows2 = build_reprsesentation(matrix2_df, n_dims, representation1)
    rows_target = build_reprsesentation(target_df, n_dims, representation1)

    print(f"len(rows1) = {len(rows1)}")
    print(f"len(rows2) = {len(rows2)}")
    print(f"len(rows_target) = {len(rows_target)}")

    for i in range(n_dims):

        row1 = rows1[i]
        row2 = rows2[i]
        row3 = rows_target[i]

        ## nu poti face asta, deoarece nu toate au acelasi numar de elemente pe linie !!
        for row in [row1, row2, row3]:
            
            if len(row) != 6:
                print(f"Skipping row {row} because it has {len(row)} elements instead of 6")
                continue  # Skip this row to prevent unpacking error

            _, val1, _, val2, _, target_val = row
            if target_val - (val1 + val2) > epsilon:
                print(f"{val1 + val2} != {target_val}")



if __name__ == '__main__':

    """
    file_path_matrix = 'data/a3.csv'
    file_path_target = 'data/b3.csv'
    
    matrix_df = import_data(file_path_matrix, reading_a=True)
    target_df = import_data(file_path_target, reading_a=False)

    gauss_seidel_solver(matrix_df, target_df, n_dims[2], max_iterations=10000, representation1=True)
    """

    # print(df.iloc[0]) ## accessing the first row

    a = 'data/a.csv'
    b = 'data/b.csv'
    sum_a_b = 'data/a_plus_b.csv'

    matrix1_df = import_data(a, reading_a=True)
    matrix2_df = import_data(b, reading_a=True)
    target_df = import_data(sum_a_b, reading_a=True)
    add_matrices(matrix1_df, matrix2_df, target_df, n_dims[4], representation1=True)