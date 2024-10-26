import torch
import torch.nn as nn

class ModelTest(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,input):
        output=input+1
        return output


model_test=ModelTest()
x=torch.tensor(1)
output=model_test(x)
print(output)