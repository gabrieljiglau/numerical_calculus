import numpy as np
from utils import *

n_dims = [10000, 20000, 30000, 80000, 2025]  # a_plus_b are n_dims = 2025 


## aici vei folosi două reprezentări, cea descrisă in temă și cea din colab (cu dicționare)
def gauss_seidel_solver1(matrix_df, target_df, n_dims, max_iterations=10000, representation1=True, precision=15):
    
    epsilon = 10 ** (-precision)
    diagonal = get_diagonal(matrix_df, n_dims)
    
    if is_diagonal_null(diagonal):
        print("Cannot perform Gauss-Seidel if any element from the diagonal is 0")
        return

    if representation1:
        rows = build_reprsesentation(matrix_df, n_dims, representation1=True)
    else:
        rows = build_representation(matrix_df, n_dims, representation1=False)

    ## make the algorithm work with the second, dictionary based, representation
    x_gs = np.zeros((n_dims, 2))
    
    k = 0
    delta_x = 1
    
    while delta_x >= epsilon and delta_x <= 10 ** 8 and k < max_iterations:
        
        for i in range(n_dims):
            
            b_i = target_df.values[i][0] 
            
            sum_aij_xij = get_sum(rows[i], x_gs, i)
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
        print(f"Residual norm = {np.max(np.sum(np.abs(residual)))}")  # A_inf
    else:
        print("Divergence!")


if __name__ == '__main__':

    file_path_matrix = 'data/a3.csv'
    file_path_target = 'data/b3.csv'
    
    matrix_df = import_data(file_path_matrix, reading_a=True)
    target_df = import_data(file_path_target, reading_a=False)

    gauss_seidel_solver1(matrix_df, target_df, n_dims[2], max_iterations=10000, representation1=True)
    # print(target_df.iloc[0])

    # print(df.iloc[0]) ## accessing the first row