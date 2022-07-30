import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from feature_selection import feature_select_pearson, xgboost_wrapper


def grid_search_cv(x_train, y_train, x_test, y_test):
    parameters = {
        'max_depth': [6],
        'n_estimators': [1000],
        'learning_rate': [0.2]
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", seed=42)

    fit_params = {
        "early_stopping_rounds": 5,
        "eval_set": [(x_test, y_test)]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring='neg_mean_squared_error',
        n_jobs=15,
        cv=2
    )

    grid.fit(x_train, y_train, **fit_params)

    # Step 3.输出网格搜索结果
    print("best_params_:")
    print(grid.best_params_)
    print('The best estimator is:')
    print(grid.best_estimator_)
    print('The score is:')
    print(np.sqrt(-grid.best_score_))
    return grid.best_estimator_


def xgb_filter(train, test):
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

    best_estimator = grid_search_cv(x_train.loc[:, x_train.columns != 'card_id'], y_train, x_test.loc[:, x_test.columns != 'card_id'], y_test)
    y_pred = best_estimator.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    # 在测试集上加入target，也就是预测标签
    x_test['predict_target'] = y_pred
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    x_test[['card_id', 'predict_target']].to_csv("../result/xgboost_filter.csv", index=False)


def xgb_wrapper(train, test):
    train, test = xgboost_wrapper(train, test)
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

    best_estimator = grid_search_cv(x_train.loc[:, x_train.columns != 'card_id'], y_train, x_test.loc[:, x_test.columns != 'card_id'], y_test)
    y_pred = best_estimator.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    # 在测试集上加入target，也就是预测标签
    x_test['predict_target'] = y_pred
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    x_test[['card_id', 'predict_target']].to_csv("../result/xgboost_wrapper.csv", index=False)


train = pd.read_csv("../preprocess/new_train.csv")
test = pd.read_csv("../preprocess/new_test.csv")

xgb_filter(train, test)
xgb_wrapper(train, test)
