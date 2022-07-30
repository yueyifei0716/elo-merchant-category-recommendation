import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/train.csv')
test=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/test.csv')
features=train.drop(['card_id','target'],axis=1).columns
train_num=train.shape[0]
test_num=test.shape[0]
def unit_value_rule(features):
    for feature in features:
        (train[feature].value_counts().sort_index() / train_num).plot()
        (test[feature].value_counts().sort_index() / test_num).plot()
        plt.legend(['train', 'test'])
        plt.xlabel(feature)
        plt.ylabel('ratio')
        plt.show()
def value_combine(feature_1,feature_2,df):
    feature1 = df[feature_1].astype(str).values.tolist()
    feature2 = df[feature_2].astype(str).values.tolist()
    return pd.Series([feature1[i]+'&'+feature2[i] for i in range(df.shape[0])])
def multiple_value_rule(features):
    for feature in features[1:]:
        train_value=(value_combine(features[0],feature,train).value_counts().sort_index()) / train_num
        test_value=(value_combine(features[0], feature, test).value_counts().sort_index()) / test_num
        index_value=pd.Series(train_value.index.tolist() + test_value.index.tolist()).drop_duplicates().sort_values()
        (index_value.map(train_value).fillna(0)).plot()
        (index_value.map(test_value).fillna(0)).plot()
        plt.legend(['train', 'test'])
        plt.xlabel('&'.join([features[0],feature]))
        plt.ylabel('ratio')
        plt.show()
unit_value_rule(features)
multiple_value_rule(features)