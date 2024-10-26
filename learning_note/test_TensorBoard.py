from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

#add_image用法
writer=SummaryWriter('../logs')
img_path= "../hymenoptera_data/train/bees/29494643_e3410f0d37.jpg"
img_PIT=Image.open(img_path)
img_array=np.array(img_PIT)
print(type(img_array))
print(img_array.shape)
writer.add_image("train",img_array,3,dataformats='HWC')
#add_scalar用法

#启动Tensorboard的方法：tensorboard --logdir=logs --port=6007
# for i in range(100):
#     writer.add_scalar("y=x*x*x",i*i*i,i)
#     #        tag (str): Data identifier   标题
#     #        scalar_value (float or string/blobname): Value to save   y值
#     #        global_step (int): Global step value to record         x值
writer.close()