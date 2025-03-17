from utils import *

n_dims = [10000, 20000, 30000, 80000, 2025]  # a_plus_b are n_dims = 2025 


## aici vei folosi două reprezentări, cea descrisă in temă și cea din colab (cu dicționare)
def gauss_seidel_solver1(matrix_df, target_df, n_dims, precision=7):

    epsilon = 10 ** (-precision)
    diagonal = get_diagonal(matrix_df, n_dims)

    diagonal_check = is_diagonal_null(diagonal)
    print(f"does the diagonal contain any null element ? {diagonal_check}")

    if diagonal_check is True:
        print("Cannot perfrom gauss_seidel if any element from the diagonal is 0")   ## division by 0
        return

    rows = build_reprsesentation1(matrix_df, n_dims)
    # print(rows[0:10])

    x_prev = [0 for _ in range(n_dims)]
    x_current = [0 for _ in range(n_dims)]

    ## print norm ||matrix_df * x_found - b||_inf (adică maximul de pe rând)

if __name__ == '__main__':

    file_path_matrix = 'data/a1.csv'
    file_path_target = 'data/b1.csv'
    
    matrix_df = import_data(file_path_matrix, reading_a=True)
    target_df = import_data(file_path_target, reading_a=False)

    gauss_seidel_solver1(matrix_df, target_df, n_dims[0])

    # print(df.iloc[0]) ## accessing the first row