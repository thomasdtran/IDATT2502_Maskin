"""
Lag en modell som predikerer tilsvarende NOT-operatoren.
Visualiser resultatet etter optimalisering av modellen.

"""

from numpy.core.fromnumeric import sort
from numpy.core.numeric import indices
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.functional import Tensor


x_train = torch.tensor([[0.0],[1.0]])
y_train = torch.tensor([[1.0],[0.0]])


class SigmoidModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.randn(1,1,requires_grad=True)
        self.b = torch.randn(1,1,requires_grad=True)
        self.m = torch.nn.Sigmoid()
    # Predictor
    def f(self, x):
        return self.m(x @ self.W + self.b)  # @ corresponds to matrix multiplication

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x),y)


model = SigmoidModel()
print(model.W)
print(x_train)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))


# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.range(torch.min(x_train), torch.max(x_train), 0.001).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = σ(xW+b)$')
plt.legend()
plt.show()
