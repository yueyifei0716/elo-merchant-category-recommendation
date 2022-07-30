import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

origian_data=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/train_pre.csv')
origian_data=origian_data.drop('card_id',axis=1)
col_num=origian_data.shape[0]
train=np.array(origian_data.iloc[:int(col_num*0.7),:])
test=np.array(origian_data.iloc[int(col_num*0.7):,:])
x_train=torch.tensor(train[:,:-1])
y_train=torch.tensor(train[:,-1])
x_test=torch.tensor(test[:,:-1])
y_test=torch.tensor(test[:,-1])


# 定义网络
class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # x 就是传入神经网络的数据
    def forward(self, x):
        x=x.to(torch.float32)
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(4, 100, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_function = torch.nn.MSELoss()
for t in range(200):
    prediction = net(x_train)
    # 预测值在前，真实值在后
    loss = loss_function(prediction, y_train)
    # 把梯度设为0
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 优化梯度
    optimizer.step()
    if t % 10 == 0:
        plt.cla()
        plt.scatter(x_train.data.numpy(), y_train.data.numpy(), c = "r", alpha = 0.5)
        plt.plot(x_train.data.numpy(), prediction.data.numpy(), "g-", lw=5)
        plt.text(0.5, 0, loss.data, fontdict={"size":20, "color":"red"})
        plt.pause(0.1)
    # entire net
    torch.save(net, "net_test.pkl")
plt.ioff()
plt.show()
