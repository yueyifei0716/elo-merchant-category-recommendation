from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from feature_selection import feature_select_pearson, random_forest_wrapper

train = pd.read_csv("../preprocess/train.csv")
test = pd.read_csv("../preprocess/test.csv")

# train, test = feature_select_pearson(train, test)
train, test = random_forest_wrapper(train, test)

# Step 1.创建网格搜索空间
print('param_grid_search')
features = train.columns.tolist()
features.remove("card_id")
features.remove("target")


x_data = train[features]
y_data = train['target']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# parameter_space = {
#     "n_estimators": [79, 80, 81],
#     "min_samples_leaf": [29, 30, 31],
#     "min_samples_split": [2, 3],
#     "max_depth": [9, 10],
#     "max_features": ["auto", 80]
# }

parameter_space = {
    "n_estimators": [80],
    "min_samples_leaf": [29],
    "min_samples_split": [2],
    "max_depth": [10],
    "max_features": [80]
}

# Step 2.执行网格搜索过程
print("Tuning hyper-parameters for mse")
# 实例化随机森林模型
clf = RandomForestRegressor(
    criterion="squared_error",
    n_jobs=15,
    random_state=22)
# 带入网格搜索
grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
grid.fit(x_train, y_train)

# Step 3.输出网格搜索结果
print("best_params_:")
print(grid.best_params_)
# means = grid.cv_results_["mean_test_score"]
# stds = grid.cv_results_["std_test_score"]
# 此处额外考虑观察交叉验证过程中不同超参数的
# for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
print('The best estimator is:')
print(grid.best_estimator_)
print('The score is:')
print(np.sqrt(-grid.best_score_))

y_pred = grid.best_estimator_.predict(x_test)
print('The RMSE is:')
print(np.sqrt(mean_squared_error(y_test, y_pred)))
# test[['card_id', 'target']].to_csv("../result/randomforest.csv", index=False)
