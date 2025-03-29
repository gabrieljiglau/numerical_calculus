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


def find_eigenvector_eigenvalue(n_dims, input_df=None, precision = 10, max_iterations=10000):

	if input_df is None:
		rows = build_symmetric_matrix(n_dims=n_dims, sparsity=0.1, max_value=20)
		print(f"Using a generated symmetric matrix")
	else:
		rows = build_reprsesentation(df, representation1=True, doing_addition=False)
		print(f"Using an already defined matrix")

	epsilon = 10 ** (-precision)
	
	if is_matrix_symmetric(rows):
		print("Input matrix is symmetric")
	else:
		print("Input matrix is not symmetric, returning ..")
		return

	current_v = np.array(generate_unit_vector(n_dims))
	current_w = np.array(get_matrix_product(rows, current_v, n_dims))
	current_lambda = np.dot(current_w, current_v)
	rayleigh_coefficient = 1
	num_iterations = 0

	## + nu este testat, deci habar n-am daca merge !! 

	## aici s-ar putea sa trebuiasca ca matricile sa fie inmultite (,nu produs scalar, intre lambda si v)
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
	a_u_max = get_matrix_product(rows, current_w)
	lambda_u_max = np.dot(current_lambda, current_w)
	print(f"norm = {np.linalg.norm(a_u_max, lambda_u_max)}")


if __name__ == '__main__':
	
	# df = import_data(file_256, reading_a=True)
	
	find_eigenvector_eigenvalue()
