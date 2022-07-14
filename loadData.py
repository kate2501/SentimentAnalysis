import os
import pandas as pd

PATH = 'data/'

class DataLoader:
    @staticmethod
    def data_load(path):
        return pd.read_pickle(os.path.join(PATH, path))

    @staticmethod
    def check(path):
        return os.path.exists(os.path.join(PATH, path))

    @staticmethod
    def save_data(df, path):
        df.to_pickle(os.path.join(PATH, path))

    @staticmethod
    def save_data_csv(df, path):
        df.to_csv(os.path.join(PATH, path))
