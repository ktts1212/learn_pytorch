import torch
import torch.nn as F

inputs= torch.tensor([1,2,3],dtype=torch.float)
targets= torch.tensor([4,5,6])
loss1=F.L1Loss()
mse_loss=F.MSELoss()
output1=loss1(inputs,targets)
output2=mse_loss(inputs,targets)
print(output1)
print(output2)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
loss_cross=F.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)