from model import Model
from loadData import DataLoader
from config import CFG
import pandas as pd
if __name__ == '__main__':
    dl = DataLoader()
    model = Model()
    train_data = dl.data_load('train.pkl')
    test_data = dl.data_load('test.pkl')

    model.run(train_data['Message'], train_data['User_Rating'],
               test_data['Message'], test_data['User_Rating'])




