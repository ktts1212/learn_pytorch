import random
import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils import data
from torch import nn

def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w)+b      #实现矩阵与矩阵，矩阵与向量之间的乘积
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))  #返回的y是一个二维张量

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

print('features:',features[0],'\nlabels:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)
d2l.plt.show()

def load_array(data_arrays,bach_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,bach_size,shuffle=is_train)

batch_size=100
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'Epoch {epoch+1}, Loss: {l.item():f}')


