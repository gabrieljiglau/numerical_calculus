import os
import pandas as pd
import gdown
from typing import List

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

def download_data(file_id, new_name, reading_a=True, data_dir="data"):

    os.makedirs(data_dir, exist_ok=True)    
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    file_path = os.path.join(data_dir, new_name)

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    gdown.download(url, output=file_path, quiet=False)
    clean_csv(file_path)

    if reading_a:
        return pd.read_csv(new_name, header=None, delimiter=r'\s*,\s*')
    else:
        return pd.read_csv(new_name, header=None)

def clean_csv(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [",".join(map(str.strip, line.split(","))) for line in lines]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(cleaned_lines))

def clean_data(df, reading_a=True):

    if reading_a:
        df.columns = ['value', 'row_idx', 'col_idx']
    else:
        df.columns = ['target']
    return df

def import_data(file_path, reading_a):
    df = pd.read_csv(file_path, header=None)
    df = clean_data(df, reading_a)
    return df

def is_diagonal_null(diagonal: List[int]):
    return 0 in diagonal

def get_diagonal(df, n_dims):

    print(f"column names = {df.columns}")
    diagonal = [0 for _ in range(n_dims)]

    diagonal_entries = df[df['row_idx'] == df['col_idx']]
    for _, row in diagonal_entries.iterrows():
        diagonal[int(row['row_idx'])] = row['value']

    return diagonal

def build_reprsesentation1(df, n_dims):

    rows = [[] for _ in range(n_dims)]
    entries = df[df['row_idx'] != df['col_idx']]  # all, but the diagonal elements
    
    for _, row in entries.iterrows():
        row_idx = int(row['row_idx'])
        col_idx = int(row['col_idx'])
        value = int(row['value'])

        rows[row_idx].append((value, col_idx))

    return rows

if __name__ == '__main__':
    pass