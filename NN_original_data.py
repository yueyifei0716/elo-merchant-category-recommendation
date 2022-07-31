import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pylab as pl

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
pd.set_option('display.max_columns', None)


origian_data = pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/train_pre.csv')
col_num = origian_data.shape[0]
origian_data = origian_data.drop('card_id', axis=1)

train = np.array(origian_data.iloc[:, :]).astype(float)
train_all = np.array(origian_data.iloc[:, :]).astype(float)
x_train = torch.tensor(train[:, :-1]).to(torch.float32)
y_train = torch.tensor(train[:, -1]).to(torch.float32)


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        x = x.squeeze(-1)
        return x


net = Net(4, 100, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_function = torch.nn.MSELoss()
loss_value=[]
for t in range(200):
    prediction = net(x_train)
    # 预测值在前，真实值在后
    RSEM_loss = torch.sqrt(loss_function(prediction, y_train))
    loss_value.append(float(RSEM_loss))
    # 把梯度设为0
    optimizer.zero_grad()
    # 反向传播
    RSEM_loss.backward()
    # 优化梯度
    optimizer.step()
print(RSEM_loss)
fig = plt.figure(figsize = (7,5))
pl.plot(range(200),loss_value,'g-',label=u'Loss Value')
# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
pl.legend()
plt.xlabel(u'iters')
plt.ylabel(u'loss value')
plt.title('loss curve')
pl.show()