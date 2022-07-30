from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import pandas as pd
import numpy as np
from feature_selection import feature_select_pearson
from sklearn.model_selection import KFold
from numpy.random import RandomState


def train_predict(train, test, best_clf):
    """
    进行训练和预测输出结果
    :param train:训练集
    :param test:测试集
    :param best_clf:最优的分类器模型
    :return:
    """

    # Step 1.选择特征
    print('train_predict...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")

    # Step 2.创建存储器
    # 测试集评分存储器
    prediction_test = 0
    # 交叉验证评分存储器
    cv_score = []
    # 验证集的预测结果
    prediction_train = pd.Series([], dtype='float64')

    # Step 3.交叉验证
    # 实例化交叉验证评估器
    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    # 执行交叉验证过程
    for train_part_index, eval_index in kf.split(train[features], train['target']):
        # 在训练集上训练模型
        best_clf.fit(train[features].loc[train_part_index], train['target'].loc[train_part_index])
        # 模型训练完成后，输出测试集上预测结果并累加至prediction_test中
        prediction_test += best_clf.predict(test[features].values)
        # 输出验证集上预测结果，eval_pre为临时变量
        eval_pre = best_clf.predict(train[features].loc[eval_index].values)
        # 输出验证集上预测结果评分，评估指标为MSE
        score = np.sqrt(mean_squared_error(train['target'].loc[eval_index].values, eval_pre))
        # 将本轮验证集上的MSE计算结果添加至cv_score列表中
        cv_score.append(score)
        print(score)
        # 将验证集上的预测结果放到prediction_train中
        # prediction_train = prediction_train.append(pd.Series(best_clf.predict(train[features].loc[eval_index]),
        #                                                      index=eval_index))
        prediction_train = pd.concat([prediction_train, pd.Series(best_clf.predict(train[features].loc[eval_index]),
                                                                  index=eval_index)])

    # 打印每轮验证集得分、5轮验证集的平均得分
    print(cv_score)
    print('The score is:')
    print(sum(cv_score) / 5)
    # 验证集上预测结果写入本地文件
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_randomforest.csv", index=False)
    # 测试集上平均得分写入本地文件
    pd.Series(prediction_test / 5).to_csv("preprocess/test_randomforest.csv", index=False)
    # 在测试集上加入target，也就是预测标签
    test['target'] = prediction_test / 5
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)
    return


train = pd.read_csv("../preprocess/train.csv")
test = pd.read_csv("../preprocess/test.csv")

train, test = feature_select_pearson(train, test)

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
# grid.fit(train[features], train['target'])

# # Step 3.输出网格搜索结果
# print("best_params_:")
# print(grid.best_params_)
# means = grid.cv_results_["mean_test_score"]
# stds = grid.cv_results_["std_test_score"]
# # 此处额外考虑观察交叉验证过程中不同超参数的
# for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print('The best estimator is:')
# print(grid.best_estimator_)
# print('The score is:')
# print(np.sqrt(-grid.best_score_))

#define cross-validation method to use
cv = KFold(n_splits=5, random_state=22, shuffle=True)

#build multiple linear regression model
# model = LinearRegression()

#use k-fold CV to evaluate model
scores = cross_val_score(grid.best_estimator_, x_train, y_train, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

#view mean absolute error
print(np.sqrt(np.mean(np.absolute(scores))))

# train_predict(train, test, grid.best_estimator_)

# test['target'] = grid.best_estimator_.predict(test[features])
# test[['card_id', 'target']].to_csv("../result/randomforest.csv", index=False)
