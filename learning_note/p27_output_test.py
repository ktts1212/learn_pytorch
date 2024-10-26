import torch
import torchvision
from PIL import Image
from torch import nn

img_path = '../imgs/plane.png'
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(img)
print(image.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load('net_gpu_29.pth')
print(model)

image = torch.reshape(image, (1, 3, 32, 32)).cuda()

model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(dim=1))
