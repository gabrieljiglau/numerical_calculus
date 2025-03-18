import numpy as np
from utils import *

n_dims = [10000, 20000, 30000, 80000, 2025]  # a_plus_b are n_dims = 2025 


## aici vei folosi două reprezentări, cea descrisă in temă și cea din colab (cu dicționare)
def gauss_seidel_solver1(matrix_df, target_df, n_dims, max_iterations=10000, representation1=True, precision=7):

    epsilon = 10 ** (-precision)
    diagonal = get_diagonal(matrix_df, n_dims)

    diagonal_check = is_diagonal_null(diagonal)
    print(f"does the diagonal contain any null element ? {diagonal_check}")

    if diagonal_check is True:
        print("Cannot perfrom gauss_seidel if any element from the diagonal is 0")   ## division by 0
        return

    if representation1:
        rows = build_reprsesentation1(matrix_df, n_dims)
    else:
        pass  ## representation2 
    # print(rows[0:10])

    x_gs = np.zeros(n_dims)

    k = 0
    delta_x = 1
    
    while delta_x >= epsilon and delta_x <= 10 ** 8 and k <= max_iterations:

        delta_x = 0

        if representation1:
            for i in range(n_dims):
                old_val = x_gs[i]

                sum_aij_xij = get_sum(rows[i], x_gs)
                b_i = target_df.values[i][0]
                new_xi = (b_i - sum_aij_xij )/ diagonal[i]

                x_gs[i] = new_xi
                diff = abs(new_xi - old_val)
                print(f"diff = {diff}")
                
                if diff > delta_x:
                    delta_x = diff
                # delta_x = max(delta_x, abs(new_xi - x_gs[i]))

        else:
            pass

        k += 1
        print(f"Currently at iteration = {k}, delta_x = {delta_x}")

    
    if delta_x < epsilon:
        print(f"found x_approx: {x_approx} = {x_current}")
        a_x = get_product(rows, x_gs, n_dims, representation1)
        norm = np.max(np.sum(np.abs(a_x - target_df)))  # # A_inf, row_max
        print(f"found_norm = {norm}")
    else:
        print(f"divergence !!")

if __name__ == '__main__':

    file_path_matrix = 'data/a1.csv'
    file_path_target = 'data/b1.csv'
    
    matrix_df = import_data(file_path_matrix, reading_a=True)
    target_df = import_data(file_path_target, reading_a=False)

    gauss_seidel_solver1(matrix_df, target_df, n_dims[0], max_iterations=100, representation1=True)
    # print(target_df.iloc[0])

    # print(df.iloc[0]) ## accessing the first row