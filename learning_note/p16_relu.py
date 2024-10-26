import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
datasets=torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataLoader=DataLoader(datasets,batch_size=64)
class Liyong(nn.Module):
    def __init__(self):
        super(Liyong, self).__init__()
        self.relu1=nn.ReLU()
        self.sigmoid1=nn.Sigmoid()

    def forward(self,x):
        output=self.sigmoid1(x)
        return output

liyong=Liyong()
writer=SummaryWriter('logs')
step=0
for data in dataLoader:
    imgs,targets=data
    writer.add_images("sigmoid_imgs",imgs,step)
    output=liyong(imgs)
    writer.add_images("sigmoid_output",output,step)
    step+=1
writer.close()