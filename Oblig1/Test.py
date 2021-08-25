import torch
import numpy as np

A = torch.tensor([[3,4,6,7,4,5]])
a = A.reshape(-1,3)
print(A.reshape(-1,3))

#Expected: 4,6 and 4,5
print(a[:,[1,2]])

