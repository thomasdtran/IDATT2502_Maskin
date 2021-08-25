import torch
import numpy as np

A = torch.tensor([[3,4,6,7,4,5]])
print(A.reshape(-1,3))