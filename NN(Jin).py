import pandas as pd
import numpy as np
from keras import models
from keras import layers

origian_data=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/train_pre.csv')
origian_data=origian_data.drop('card_id',axis=1)
col_num=origian_data.shape[0]
train=np.array(origian_data.iloc[:int(col_num*0.7),:])
test=np.array(origian_data.iloc[int(col_num*0.7):,:])
print(train)
x_train=train[:,:-1]
y_train=train[:,-1]
x_test=test[:,:-1]
y_test=test[:,-1]


# 构建神经网络
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # !网络的最后一层只有一个单元,没有激活,是一个线性层
    # !这是标量回归（标量回归是预测单一连续值的回归）的典型设置
    model.add(layers.Dense(1))
    # !编译网络用的是mse损失函数,即均方误差（MSE, mean squared error）
    # !预测值与目标值之差的平方,这是回归问题常用的损失函数

    # !平均绝对误差（MAE, mean absolute error）
    # !是预测值与目标值之差的绝对值
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K折验证

k = 5
num_value_sample = len(x_train) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("Processing fold #", i)
    # !提取第i折的训练集与验证集
    val_data = x_train[i * num_value_sample: (i + 1) * num_value_sample]
    val_targets = y_train[i * num_value_sample: \
                                (i + 1) * num_value_sample]

    partial_train_data = np.concatenate(
        [x_train[:i * num_value_sample],
         x_train[(i + 1) * num_value_sample:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_value_sample],
         y_train[(i + 1) * num_value_sample:]],
        axis=0)

    # !使用训练集训练
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)

    # !使用验证集验证
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
