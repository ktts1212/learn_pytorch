import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor())

dataLoder=DataLoader(dataset,batch_size=64,drop_last=True)

class Liyong(nn.Module):
    #这里用来定义层，将神经网络的不同部分封装到一起
    def __init__(self):
        super(Liyong, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    #定义前向传播
    def forward(self,x):
        x=self.conv1(x)
        return x

liong=Liyong()
writer=SummaryWriter(log_dir='logs')

step=0
for data in dataLoder:
    imgs,tragets=data
    output=liong(imgs)
    # print(imgs.shape)
    # print(output.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    #torch.Size([64, 6, 30, 30])
    #没有下面这行代码会报错，因为图像rgb为3通道，6通道的话summarywriter不知道如何显示
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step+=1

writer.close()