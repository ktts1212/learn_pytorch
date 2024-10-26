from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")

img=Image.open("../hymenoptera_data/train/bees/95238259_98470c5b10.jpg")
print(img)

#toTensor
trans_toTensor=transforms.ToTensor()
img_tensor=trans_toTensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize
print(img_tensor.size())
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,3,2],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)


#Resize
print(img.size)
trans_resize=transforms.Resize(255)
img_resize=trans_resize(img)
img_resize=trans_toTensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize.size())
print(img_tensor.size())

#Compose
trans_resize_2=transforms.Resize(255)
#PIL->PIL->tensor
trans_compose=transforms.Compose([trans_resize_2,trans_toTensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop
trans_random=transforms.RandomCrop(128)
trans_compose_2=transforms.Compose([trans_random,trans_toTensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()
