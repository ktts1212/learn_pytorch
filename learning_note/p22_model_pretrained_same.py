import torch
import torchvision

vgg16_false=torchvision.models.vgg16(pretrained=False)
#保存方式1 torch.save,保存模型结构及参数，
torch.save(vgg16_false,"vgg16_fn")
print(vgg16_false)
#方式1加载模型方法
#通过torch.load("vgg16_fn")获取模型
vgg16_load1=torch.load("vgg16_fn")

#保存方式2,保存为字典形式，保存模型参数
torch.save(vgg16_false.state_dict(),"vgg16_fn2")
#方式2加载模型的方法
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_fn2"))