import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from feature_selection import feature_select_pearson, random_forest_wrapper


def grid_search_cv(x_train, y_train):
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
        random_state=42)
    # 带入网格搜索
    grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(x_train, y_train)

    # Step 3.输出网格搜索结果
    print("best_params_:")
    print(grid.best_params_)
    print('The best estimator is:')
    print(grid.best_estimator_)
    print('The score is:')
    print(np.sqrt(-grid.best_score_))
    return grid.best_estimator_


def rf_filter(train, test):
    train, test = feature_select_pearson(train, test)
    # Step 1.创建网格搜索空间
    print('param_grid_search')
    features = train.columns.tolist()
    # features.remove("card_id")
    features.remove("target")
    # x_data = train[features]
    # y_data = train['target']
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

    best_estimator = grid_search_cv(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = best_estimator.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE on validation set is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    # prediction_test = best_estimator.predict(test[features])
    # 在测试集上加入target，也就是预测标签
    x_test['predict_target'] = y_pred
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    x_test[['card_id', 'predict_target']].to_csv("../result/randomforest_filter.csv", index=False)


def rf_wrapper(train, test):
    train, test = random_forest_wrapper(train, test)
    # Step 1.创建网格搜索空间
    print('param_grid_search')
    features = train.columns.tolist()
    # features.remove("card_id")
    features.remove("target")
    # x_data = train[features]
    # y_data = train['target']
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

    best_estimator = grid_search_cv(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = best_estimator.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    # prediction_test = best_estimator.predict(test[features])
    # 在测试集上加入target，也就是预测标签
    x_test['predict_target'] = y_pred
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    x_test[['card_id', 'predict_target']].to_csv("../result/randomforest_wrapper.csv", index=False)


train = pd.read_csv("../preprocess/new_train.csv")
test = pd.read_csv("../preprocess/new_test.csv")

rf_filter(train, test)
rf_wrapper(train, test)
