import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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

liyong=Liyong()
print(liyong)
input=torch.ones(64,3,32,32)
output=liyong(input)
print(output.shape)

writer=SummaryWriter(log_dir='logs')
writer.add_graph(liyong,input)
writer.close()