from utils import *

a1_file_id = '1DPNqhir87-JyFXzq0JZ6PzDRnzARcH8M'
a2_file_id='1NNbu5IfIlFurroNTMlg6NzQ_5Bww4H_m'
a3_file_id = '1hc25jCkzH2gefuSmvoU370ggbytYCLRa'
a4_file_id = '1AZX2nXjfshYSXpK7Td4qSFxTnBa8HDSn'
a5_file_id = '1Xb8nBaS6aNh4HN0avfK8eOD4iBXu-53G'

b1_file_id = '1EVx6pniErrP6Gkdd0mdo_rzOHsEW7Uul'
b2_file_id = '1bPSnby7y200XSHFkJlvHKHRwQRBY6Xqe'
b3_file_id = '1rW72eaWeodzhZ1VxGKKoVU93zS7p2Etp'
b4_file_id = '1UWIMLa9_P4MgQpRSZvEIKgdgG7Ui28LA'
b5_file_id = '1QI-ChYZRjPbGmV-su0kaMlYA_WA9KLx3'

a_file_id = '16moWqNDr2g7qT7bIGnjH3Q5YR0cJ7hr-'
b_file_id = '1vapIlKJL7Lb8-OgoJQCNZW-4DJWo9COi'
a_plus_b = '1ybOOPQOQJCWVYBzWFwFNMw_iD3Xl8dGV'

n_dims = [10000, 20000, 30000, 80000, 2025]

if __name__ == '__main__':

    """
    trebuie curățate și restul fișierelor de input
    """

    df = import_data(a1_file_id, "a1.csv", reading_a=True)
    df = clean_data(df, reading_a=True)

    get_diagonal(df, n_dims[0])
