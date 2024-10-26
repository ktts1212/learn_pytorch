import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

datasets=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor())

dataLoader=DataLoader(datasets, batch_size=1, shuffle=True)
class Liyong(nn.Module):
    def __init__(self):
        super(Liyong, self).__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        output=self.seq(x)
        return output

loss=nn.CrossEntropyLoss()
liyong=Liyong()
optim=torch.optim.SGD(liyong.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0
    for data in dataLoader:
        imgs,targets=data
        output=liyong(imgs)
        result_loss=loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss
    print(running_loss)