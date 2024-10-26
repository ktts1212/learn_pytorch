import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

dataLoader=DataLoader(dataset,batch_size=64)
# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]
#                     ])
#
# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Liyong(nn.Module):
    def __init__(self):
        super(Liyong, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        output=self.maxpool1(x)
        return output

liyong=Liyong()
# output=liyong(input)
# print(output)
step=0
writer=SummaryWriter(log_dir='logs')
for data in dataLoader:
    imgs,targets=data
    writer.add_images("inputMaxPool",imgs,step)
    output=liyong(imgs)
    writer.add_images("outputMaxPool",output,step)
    step+=1

writer.close()

