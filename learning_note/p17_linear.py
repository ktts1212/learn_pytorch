import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

datasets=torchvision.datasets.CIFAR10(root='./data', train=False,transform=torchvision.transforms.ToTensor(), download=True)
dataLoader=DataLoader(datasets,batch_size=64,drop_last=True)

class Liyong(nn.Module):
    def __init__(self):
        super(Liyong, self).__init__()
        self.linear1=nn.Linear(196608,10)

    def forward(self,x):
        output=self.linear1(x)
        return output

liyong=Liyong()
for data in dataLoader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.flatten(imgs)
    print(output.shape)
    output=liyong(output)
    print(output.shape)