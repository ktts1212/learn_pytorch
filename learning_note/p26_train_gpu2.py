import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#定义训练的设备
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_len = len(train_data)
test_data_len = len(test_data)
print(f"训练集长度为:{train_data_len}")
print(f"测试集长度为:{test_data_len}")

# 加载数据
train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

#创建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x=self.model(x)
        return x
# 加载神经网络
net = Net()
#使用gpu进行训练
net.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 30
#添加tensorboard
writer=SummaryWriter(log_dir='logs')

for i in range(epoch):
    print(f"-------第{i + 1}轮训练开始-------")

    #训练步骤开始
    net.train() #在某些模型下必须调用该方法
    for data in train_dataLoader:
        imgs, targets = data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤
    net.eval()
    total_test_loss = 0
    total_accuarcy=0

    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuarcy+=accuracy

    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuarcy/test_data_len}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuarcy/test_data_len,total_test_step)
    total_test_step+=1

    #保存每一轮训练结果
    torch.save(net,"net_gpu_{}.pth".format(i))
    print("模型已保存")

writer.close()