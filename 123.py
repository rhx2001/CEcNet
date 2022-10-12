import torch
a=torch.tensor([1,2,3,4,5,10])
num=torch.randperm(3)
a[3:][num]
a[:3][num]
