"""
Lag en modell som predikerer tilsvarende XOR-operatoren. Før
du optimaliserer denne modellen må du initialisere
modellvariablene med tilfeldige tall for eksempel mellom -1 og
1. Visualiser både når optimaliseringen konvergerer og ikke
konvergerer mot en riktig modell.

"""

from numpy.core.fromnumeric import sort
from numpy.core.numeric import indices
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.functional import Tensor
import random


x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.double).reshape(-1, 2)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.double).reshape(-1, 1)


class SigmoidModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
  
        """self.W1 = torch.randn(2, 2, requires_grad=True) 
        self.b1 = torch.randn(1, 2, requires_grad=True)
        self.W2 = torch.randn(2, 1, requires_grad=True)
        self.b2 = torch.randn(1, 1, requires_grad=True)"""

        self.W1 = torch.tensor([[2, -4], [10, -10]], dtype=torch.double, requires_grad=True)
        self.b1 = torch.tensor([[-5,15]], dtype=torch.double, requires_grad=True)
        self.W2 = torch.tensor([[10], [10]], dtype=torch.double, requires_grad=True)
        self.b2 = torch.tensor([-15], dtype=torch.double, requires_grad=True)

        self.m = torch.nn.Sigmoid()

    #first layer
    def f1(self, x):
        # @ corresponds to matrix multiplication
        return self.m(x @ self.W1 + self.b1)

    #second layer
    def f2(self, x):
        # @ corresponds to matrix multiplication
        return self.m(x @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=0.1)
for epoch in range(50000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %
      (model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train)))


#Plots the data as well as the plane for the model
ax = plt.axes(projection="3d")

ax.scatter3D(x_train[:, 0], x_train[:, 1], y_train)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")


x1 = torch.arange(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 0.001, dtype=torch.double)
x2 = torch.arange(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 0.001, dtype=torch.double)

X1, X2 = torch.meshgrid(x1, x2)

#y = predicted output
y = model.f(torch.cat((torch.ravel(X1).unsqueeze(1), torch.ravel(
    X2).unsqueeze(1)), dim=-1).type(torch.double))
Y = y.reshape(X1.shape).detach()

ax.plot_wireframe(X1, X2, Y)
plt.show()
