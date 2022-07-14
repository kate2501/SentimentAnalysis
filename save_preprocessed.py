from loadData import DataLoader
import pandas as pd
from model import Model

dl = DataLoader()
model = Model()
train_data = dl.data_load('bank_reviews_train.pkl')
test_data = dl.data_load('bank_reviews_test.pkl')
X_train, y_train = model.preprocess(train_data['Message'], train_data['User_Rating'])
X_test, y_test = model.preprocess(test_data['Message'], test_data['User_Rating'])
dl.save_data(pd.concat([X_train, y_train], axis=1), 'train.pkl')
dl.save_data(pd.concat([X_test, y_test], axis=1), 'test.pkl')

