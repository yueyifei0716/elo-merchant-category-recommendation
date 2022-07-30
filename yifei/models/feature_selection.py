import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb


def feature_select_pearson(train, test):
    """
    利用pearson系数进行相关性特征选择
    :param train:训练集
    :param test:测试集
    :return:经过特征选择后的训练集与测试集
    """
    print('feature_select...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    featureSelect = features[:]

    # 去掉缺失值比例超过0.99的
    for fea in features:
        if train[fea].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(fea)

    # 进行pearson相关性计算
    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

    # 取top300的特征进行建模，具体数量可选
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = ['card_id'] + se[:300].index.tolist()
    print('done')
    return train[feature_select + ['target']], test[feature_select + ['target']]


def xgboost_wrapper(train, test):
    """
    lgm特征重要性筛选函数
    :param train:训练数据集
    :param test:测试数据集
    :return:特征筛选后的训练集和测试集
    """

    # Part 1.划分特征名称，删除ID列和标签列
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    params_initial = {
        'max_depth': 6,
        'n_estimators': 1000,
        'learning_rate': 0.2,
        'objective': "reg:squarederror",
        'seed': 42
    }
    # Part 3.交叉验证过程
    # 实例化评估器
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    # 创建空容器
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        clf = xgb.XGBRegressor(**params_initial)
        clf.fit(train[features].loc[train_part_index], train[label].loc[train_part_index], early_stopping_rounds=5,
                eval_set=[(train[features].loc[eval_index], train[label].loc[eval_index])])
        fse += pd.Series(clf.feature_importances_, features)

    # Part 4.选择最重要的300个特征
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select + ['target']]


def random_forest_wrapper(train, test):
    """
    lgm特征重要性筛选函数
    :param train:训练数据集
    :param test:测试数据集
    :return:特征筛选后的训练集和测试集
    """

    # Part 1.划分特征名称，删除ID列和标签列
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    params_initial = {
        "n_estimators": 80,
        "min_samples_leaf": 29,
        "min_samples_split": 2,
        "max_depth": 10,
        "max_features": 80,
        'criterion': "squared_error",
        'n_jobs': 15,
        'random_state': 42
    }
    # Part 3.交叉验证过程
    # 实例化评估器
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    # 创建空容器
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        clf = RandomForestRegressor(**params_initial)
        clf.fit(train[features].loc[train_part_index], train[label].loc[train_part_index])
        fse += pd.Series(clf.feature_importances_, features)

    # Part 4.选择最重要的300个特征
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select + ['target']]


def lightGBM_wrapper(train, test):
    """
    lgm特征重要性筛选函数
    :param train:训练数据集
    :param test:测试数据集
    :return:特征筛选后的训练集和测试集
    """

    # Part 1.划分特征名称，删除ID列和标签列
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # Step 2.配置lgb参数
    # 模型参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'min_child_samples': 20,
        'bagging_seed': 2020,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'metric': 'rmse',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'regression'
    }
    # 控制参数
    # 提前验证迭代效果或停止
    ESR = 30
    # 迭代次数
    NBR = 10000
    # 打印间隔
    VBE = 50

    # Part 3.交叉验证过程
    # 实例化评估器
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    # 创建空容器
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 封装训练数据集
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        # 封装验证数据集
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        # 在训练集上进行训练，并同时进行验证
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        # 输出特征重要性计算结果，并进行累加
        fse += pd.Series(bst.feature_importance(), features)

    # Part 4.选择最重要的300个特征
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select + ['target']]
