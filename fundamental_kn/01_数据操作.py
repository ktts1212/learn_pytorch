import torch
import os
import pandas as pd
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import numpy as np

# x=torch.arange(12)
# print(x.shape)
# x1=x.reshape(3,4)
# print(x1)
# print(x1.shape)
# x2=torch.zeros((2,3,4))
# print(x2)
# x3=torch.randn(5)
# print(x3)
# x4=torch.rand(5)
# print(x4)
# x5=torch.tensor([[2,4,5],[1,5,3]])
# print(x5)

# 两个张量链接起来
# x=torch.arange(12).reshape(3,4)
# print(x)
# y=torch.tensor([[2,3,4,5],[6,7,8,9],[11,23,11,22]])
# print(y)
# print(torch.cat((x,y),dim=1))  #dim就类似与numpy中的axis
#
# print(x==y)
# print(x.sum())
# torch中的广播机制
# a=torch.arange(12).reshape(3,4)
# print(a)
# b=torch.arange(4).reshape(1,4)
# print(b)
# #print(a+b)
#
# #节省内存
# before=id(a)
# a[:]=a+b #or a+=b
# print(before==id(a))

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
#
# data = pd.read_csv(data_file)
# print(data)
#
# inputs: pd.DataFrame = data.iloc[:, 0:2]  # tpye:pd.DataFrame
# outputs = data.iloc[:, 2]
# print(inputs)
# # print(outputs)
# inputs = inputs.fillna(inputs.mean(numeric_only=True))
# # print(inputs)
# inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

# 降为求和
# X=torch.arange(12 ,dtype=torch.float32).reshape(3,4)
# print(X)
# print(X.sum())
# print(X.mean())   #整个张量的平均值
# print(X.mean(dim=1))
# print(X.numel())  #张量中元素个数

# 非降维求和
# X=torch.arange(12).reshape(3,4)
# sum_A=X.sum(dim=0,keepdim=True)  #keepdim为False时，sum后生成的是一维张量，为True时，可以继续保持二维张量
# print(sum_A)
# print(sum_A.size())
# print(X/sum_A)

# 点积
# X=torch.tensor([0,1,2,3],dtype=torch.float)
# Y=torch.ones(4)
# print(X.dot(Y))
# print(torch.dot(X,Y)) #二者效果一样

# 矩阵与向量
# X = torch.arange(12, dtype=torch.float).reshape(3, 4)
# Y = torch.arange(4, dtype=torch.float)
# print(X)
# print(Y)
# print(torch.mv(X, Y))
# 矩阵相乘
# Y = torch.arange(20, dtype=torch.float).reshape(4, 5)
# print(torch.mm(X, Y))

# #向量的范数 包括L1范数，L2范数和Frobenius范数
# X=torch.tensor([1.0,-2.0,3.0,-4.0])
# #L1范数：取向量中所有元素绝对值的和
# print(torch.norm(X,p=1))
# #L2范数：取向量中所有元素平方和在开方
# print(torch.norm(X,p=2))
# #Frobenius范数:是矩阵型向量的L2范数
# Y=torch.ones((4,9))
# print(torch.norm(Y))

# 微分与自动微分
# def f(x):
#     return 3 * x ** 2 - 4 * x
#
#
# def numercial_lim(f, x, h):
#     return (f(x + h) - f(x)) / h
#
#
# h = 0.1
# for i in range(5):
#     print(f'h={h:.5f},numercial limit={numercial_lim(f, 1, h):.5f}')
#     h *= 0.1
#
# def use_svg_display():
#     backend_inline.set_matplotlib_formats('svg')
#
# def set_figsize(figsize=(3.5,2.5)):
#     use_svg_display()
#     plt.rcParams['figure.figsize'] = figsize
#
# def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()
#
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []
#
#     set_figsize(figsize)
#     axes = axes if axes else plt.gca()
#
#     # 如果X有一个轴，输出True
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
#                 and not hasattr(X[0], "__len__"))
#
#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# plt.show()

# 自动微分
# 这个例子为标量对向量求导，结果与向量的形状相同
# x=torch.arange(4.0)
# x.requires_grad_(True)
# y=2*torch.dot(x,x)
# y.backward() #反向传递计算倒数，结果存在x.grad
# #print(x.grad)
#
# #print(x.grad==4*x)
#
# x.grad.zero_() #在默认情况下，pytorch会累计梯度，因此需要清楚之前的值
# y=x.sum()
# #print(y)
# y.backward()
# #print(x.grad)
#
# #向量y对向量x求导:结果获得一个高阶张量
# #这个例子为隐式创建
# x.grad.zero_()
# y=x*x
# y.sum().backward()   #如果是一个标量对一个向量求导，叫梯度创建的隐式创建，如果是一个向量或张量对一个向量求导，则叫显示创建
# #print(x.grad)

# 显示创建
# x = torch.arange(4.0, requires_grad=True)
# y = x * x
# grad_tensor = torch.tensor([1.0, 1.0, 2.0, 2.0])
# y.backward(grad_tensor)
# # print(x.grad)
#
# # 分离计算
# x.grad.zero_()
# y = x * x
# u = y.detach()  # 创建一个与y相同的张量，但在梯度计算中，不参与计算，而是被视为一个常熟
# z = u * x
# z.sum().backward()
# print(x.grad)
# print(x.grad == u)

#python控制流的梯度计算
def f(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c

a=torch.randn(size=(),requires_grad=True)
print(a)
d=f(a)
print(d)
d.backward()
print(a.grad)
print(a.grad==d/a)