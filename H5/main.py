import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../H3')))
from utils import download_data, clean_csv, import_data, build_reprsesentation
from utils5 import *

# https://colab.research.google.com/drive/1NUV0hFtPK5nUPO9U1MvMEB2j_OiMNR9_#scrollTo=e5f0a116 # aici gasesti ajutor pentru tema 5

n_dims = [256, 512, 1024, 1536, 2025, 2200, 3600, 40000]

file_4000 = 'data/marar_4000.csv'
file_3600 = 'data/marar_3600.csv'
file_2200 = 'data/marar_2200.csv'
file_2025 = 'data/marar_2025.csv'
file_1536 = 'data/marar_1536.csv'
file_1024 = 'data/marar_1024.csv'
file_512  = 'data/marar_512.csv'
file_256  = 'data/marar_256.csv'


def find_eigenvector_eigenvalue(n_dims, input_df=None, precision = 12, max_iterations=100000):

	if input_df is None:
		rows = build_symmetric_matrix(n_dims=n_dims, sparsity=0.1, max_value=20)
		print(f"Using a generated symmetric matrix")
	else:
		rows = build_reprsesentation(df, n_dims=n_dims, representation1=True, doing_addition=False)
		print(f"Using an already defined matrix")

	epsilon = 10 ** (-precision)
	
	if is_matrix_symmetric(rows, epsilon):
		print("Input matrix is symmetric")
	else:
		print("Input matrix is not symmetric, returning ..")
		return

	current_v = np.array(generate_unit_vector(n_dims))
	current_w = np.array(get_matrix_product(rows, current_v, n_dims))
	current_lambda = np.dot(current_w, current_v)
	rayleigh_coefficient = 1
	num_iterations = 0

	while num_iterations < max_iterations and np.linalg.norm(current_w - np.dot(current_lambda, current_v)) >= n_dims * epsilon:
		
		pre_dot_v = current_v
		current_v = np.dot((1 / np.linalg.norm(current_w)), current_w)
		current_w = np.array(get_matrix_product(rows, current_v, n_dims))
		current_lambda = np.dot(current_w, current_v)
		
		rayleigh_coefficient = np.dot(current_v, pre_dot_v) / np.linalg.norm(pre_dot_v)

		num_iterations += 1

	if num_iterations > max_iterations:
		print("Couldn't calculate the eigenvector and eigenvalue")
		print(f"num_iterations = {num_iterations}; max_iterations = {max_iterations}")
		return
	
	print(f"Eigenvector = {current_w} \neigenvalue = {rayleigh_coefficient}")
	a_u_max = get_matrix_product(rows, current_w, n_dims)
	lambda_u_max = np.dot(current_lambda, current_w)
	print(f"norm = {np.linalg.norm(a_u_max - lambda_u_max)}")
	print(f"result found after {num_iterations + 1} iterations")


def svd_analysis(n_dim, p_dim, target, precision=12):

    epsilon = 10 ** (-precision)

    A = np.random.randn(n_dim, p_dim)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print("Valori singulare din A:", S)

    # rangul matricii A, numarul de valori singulare pozitive
    rank_A = np.sum(S > epsilon) 
    print("Rangul lui A:", rank_A)

    # numarul de conditionale, raportul dintre cea mai mare si cea mai mica valoare singulara
    sigma_max = S[0]
    sigma_min = S[rank_A - 1]  # The smallest nonzero singular value
    condition_number = sigma_max / sigma_min
    print("Condition number of A:", condition_number)

    # pseudoinversa Moore-Penrose
    S_inv = np.diag(1 / S[:rank_A])  # Inverting only nonzero singular values
    A_pseudo_inv = np.dot(Vt.T[:, :rank_A], np.dot(S_inv, U.T[:rank_A, :]))
    print("Moore-Penrose Pseudoinverse of A:\n", A_pseudo_inv)

    x_I = np.dot(A_pseudo_inv, target)
    print("Solution x_I:", x_I)

    residual_norm = np.linalg.norm(target - np.dot(A, x_I))
    print("Residual norm ||b - Ax_I||:", residual_norm)

    return A, A_pseudo_inv, x_I, residual_norm


if __name__ == '__main__':
	
	"""
	df = import_data(file_4000, reading_a=True)
	n_dims = 1234
	find_eigenvector_eigenvalue(n_dims)
	"""
	# print(f"{0.1} + {0.2} = {0.1 + 0.2}")

	n_dim = 7
	p_dim = 5
	target = np.random.randn(n_dim)
	svd_analysis(n_dim, p_dim, target)
