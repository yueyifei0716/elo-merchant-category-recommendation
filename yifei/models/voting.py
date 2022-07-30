import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


data = pd.read_csv("../result/randomforest_wrapper.csv")
data['randomforest'] = data['predict_target'].values

temp = pd.read_csv("../result/lightgbm_wrapper.csv")
data['lightgbm'] = temp['predict_target'].values

temp = pd.read_csv("../result/xgboost_wrapper.csv")
data['xgboost'] = temp['predict_target'].values

print(data.corr())

y_test = pd.read_csv("../preprocess/new_test.csv")['target']

data['voting_target_avg'] = (data['randomforest'] + data['lightgbm'] + data['xgboost']) / 3
data[['card_id', 'voting_target_avg']].to_csv("../result/voting_avg.csv", index=False)
y_pred = data['voting_target_avg']

print('The RMSE is:')
print(np.sqrt(mean_squared_error(y_test, y_pred)))

data['voting_target_weight'] = data['randomforest'] * 0.3 + data['lightgbm'] * 0.5 + data['xgboost'] * 0.2
data[['card_id', 'voting_target_weight']].to_csv("../result/voting_weight.csv", index=False)
y_pred = data['voting_target_weight']

print('The RMSE is:')
print(np.sqrt(mean_squared_error(y_test, y_pred)))
