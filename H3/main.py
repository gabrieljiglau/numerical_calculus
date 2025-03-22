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

    rows1 = build_reprsesentation(matrix1_df, n_dims, representation1, doing_addition=True)
    rows2 = build_reprsesentation(matrix2_df, n_dims, representation1, doing_addition=True)
    rows_target = build_reprsesentation(target_df, n_dims, representation1, doing_addition=True)

    if representation1:
        ## sorting for easier manipulation
        rows1 = [sorted(row, key=lambda x:x[0]) for row in rows1] # x[0] = the col_idx
        rows2 = [sorted(row, key=lambda x:x[0]) for row in rows2]
        rows_target = [sorted(row, key=lambda x:x[0]) for row in rows_target]   
    
    ## addition in itself 
    for i in range(n_dims):
        if representation1:
            rows1[i] = merge_sparse_rows(rows1[i], rows2[i])  # merges a list
        else:
            for col_idx, val in rows2[i].items():
                rows1[i][col_idx] = rows1[i].get(col_idx, 0) + val  # merging a dictionary


    if representation1:
        rows1 = [[(col, val) for col, val in row if not (col == 0 and val == 0)] for row in rows1]
        print(f"rows1[0:1] = {rows1[0:2]}")
        print(f"rows_target[0:1] = {rows_target[0:2]}")

    ## checking for equality between the matrices
    for i in range(n_dims):
        
        if representation1:
            row1 = rows1[i] 
            row3 = rows_target[i]  #

            p1, p3 = 0, 0 

            while p1 < len(row1) and p3 < len(row3):
                col_idx1, val1 = row1[p1]
                col_target, val_target = row3[p3]

                if col_idx1 == col_target:  
                    if abs(val_target - val1) > epsilon:
                        print(f"Mismatch at row {i}, column {col_idx1}: {val1} vs {val_target}")
                        exit()
                    p1 += 1
                    p3 += 1

                elif col_idx1 < col_target:  
                    print(f"Extra column: {col_idx1} in computed row {i}, that is not in target")
                    exit()
                    p1 += 1

                else:  # Extra column in row3 (target)
                    print(f"Missing column: {col_target} in computed row {i}")
                    exit()
                    p3 += 1

            # remaining elements in either row
            if p1 < len(row1):
                print(f"Extra columns in computed row {i}: {[col for col, _ in row1[p1:]]}")
                exit()
            if p3 < len(row3):
                print(f"Missing columns in computed row {i}: {[col for col, _ in row3[p3:]]}")
                exit()
        else:
            dict1 = rows1[i]
            dict2 = rows_target[i]

            all_keys = set(dict1.keys()).union(set(dict2.keys()))

            for key in all_keys:
                val1 = dict1.get(key, 0)  # value from dict, default to 0 if missing
                val2 = dict2.get(key, 0)

                if abs(val1 - val2) > epsilon:
                    print(f"Mismatch at row {i}, column {key}: {val1} vs {val2}")
                    exit()

    print("The two matrices match !!")



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
    add_matrices(matrix1_df, matrix2_df, target_df, n_dims[4], representation1=False)