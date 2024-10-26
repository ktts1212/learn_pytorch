import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.CIFAR10(root='./datasets',train=True,transform=dataset_transforms,download=True)
test_set=torchvision.datasets.CIFAR10(root='./datasets',train=False,transform=dataset_transforms,download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
#
# img.show()
print(test_set[0])
writer=SummaryWriter('logs')
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()