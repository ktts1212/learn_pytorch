import torch
outputs=torch.tensor([[0.1,0.2],
                      [0.3,0.4]])

#argmax: dim=0，竖着看，dim=1，横着看
print(outputs.argmax(dim=1))
preds=outputs.argmax(1)
targets=torch.tensor([0,1])
#targets与preds中值相同元素的个数
print((targets==preds).sum())