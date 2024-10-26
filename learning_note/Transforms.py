from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
#transform就像一个工具箱，里面的类就是工具
img_path="../hymenoptera_data/train/ants/49375974_e28ba6f17e.jpg"
img=Image.open(img_path)
print(img)
#1.Transforms如何使用
#需要创建“具体的工具”
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
writer=SummaryWriter(log_dir='logs')
writer.add_image('tensor_img',tensor_img)
writer.close()