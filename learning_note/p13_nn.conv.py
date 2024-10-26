import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]]).reshape((1,1,5,5))

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]]).reshape((1,1,3,3))
print(input.shape)
print(kernel.shape)
#关于input和kernel的shape参数
#对于输入，需要 (batch_size, in_channels, height, width) 形式。
#对于卷积核，需要 (out_channels, in_channels, kernel_height, kernel_width) 形式。
#对于输出通道output kernel 输出的张量形状可能是(batch_size,k,new_height,new_width)
# k的大小取决于有几个核(滤波器),每个核会对输入的所有通道进行卷积操作，多个核生成多个特征图，k的大小与卷积核的out_channels相同？？
#所以input，kernel，output都是四维的
output=F.conv2d(input,kernel,stride=1)
print(output)
# output2=F.conv2d(input,kernel,stride=2)
# print(output2)
# output3=F.conv2d(input,kernel,stride=1,padding=1)
# print(output3)
print(output.size())