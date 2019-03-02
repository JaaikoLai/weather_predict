
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F     # 激励函数都在这
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden2(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x


if __name__ == "__main__":
    device = torch.device(f'cuda:{0}')
    print(device)
    with torch.cuda.device(0):
        net = Net(n_feature=1, n_hidden1=50, n_hidden2=50, n_output=1)
        net.to(device)
        # if torch.cuda.is_available():
        #     net.cuda()
        print(net)  # net 的结构
        """
        Net (
        (hidden): Linear (1 -> 10)
        (predict): Linear (10 -> 1)
        )
        """


        # optimizer 是训练的工具
        optimizer = torch.optim.Adam(net.parameters(), lr=0.025)  # 传入 net 的所有参数, 学习率
        loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

        # # 先转换成 torch 能识别的 Dataset
        x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
        y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

        torch_dataset = Data.TensorDataset(x, y)

        # 把 dataset 放入 DataLoader
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=20,               # mini batch size
            shuffle=True,               # 要不要打乱数据 (打乱比较好)
            num_workers=2,              # 多线程来读数据
        )

        for t in range(50):
            for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
                # 假设这里就是你训练的地方...
                #batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_prediction = net(batch_x)     # 喂给 net 训练数据 x, 输出预测值

                loss = loss_func(batch_prediction, batch_y)     # 计算两者的误差

                optimizer.zero_grad()   # 清空上一步的残余更新参数值
                loss.backward()         # 误差反向传播, 计算参数更新值
                optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
                # 接着上面来
                prediction = net(x.to(device)).cpu()
                loss = loss.cpu()
                if step % 1 == 0:
                    # plot and show learning process
                    plt.cla()
                    plt.scatter(x.data.numpy(), y.data.numpy())
                    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
                    plt.pause(0.1)
        plt.show()