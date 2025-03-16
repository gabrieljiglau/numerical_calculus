import os
import pandas as pd
import gdown


def clean_csv(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [",".join(map(str.strip, line.split(","))) for line in lines]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(cleaned_lines))


def import_data(file_id, new_name, reading_a=True, data_dir="data"):

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

def clean_data(df, reading_a=True):

    if reading_a:
        df.columns = ['value', 'row_idx', 'col_idx']
    else:
        df.columns = ['target']
    return df

def get_diagonal(df, n_dims):

    print(f"column names = {df.columns}")
    diagonal = [0 for _ in range(n_dims)]

    diagonal_entries = df[df['row_idx'] == df['col_idx']]
    for _, row in diagonal_entries.iterrows():
        diagonal[int(row['row_idx'])] = row['value']

    print(diagonal)
    return diagonal

if __name__ == '__main__':
    pass