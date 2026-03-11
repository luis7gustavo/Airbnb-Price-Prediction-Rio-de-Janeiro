import pandas as pd

def load_dataset(path):

    df = pd.read_csv(path, compression="gzip")

    return df