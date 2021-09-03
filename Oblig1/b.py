"""
Linear regresjon i 3 dimensjoner:

Lag en linear modell som predikerer alder (i dager) ut fra lengde og vekt gitt
observasjonene i day_length_weight.txt
"""

from numpy.core.fromnumeric import sort
from numpy.core.numeric import indices
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.functional import Tensor

text = open("Oblig1/day_length_weight.txt", "r")
text.readline()

length_weight = []  # Length and weight
days = []  # age in days
for line in text:
    l = line.split(",")
    days.append(float(l[0]))
    length_weight.append(float(l[1]))
    length_weight.append(float(l[2]))

text.close()

n = len(days)  # number of observations
nt = 10  # number of tests

# Observed/training input and output
x_train = torch.tensor(length_weight).reshape(-1, 2)
y_train = torch.tensor(days).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.randn(2, 1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(500000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))


#Plots the data as well as the plane for the model
ax = plt.axes(projection="3d")

ax.scatter3D(x_train[:, 0], x_train[:, 1], y_train)

ax.set_xlabel("length")
ax.set_ylabel("weight")
ax.set_zlabel("days")

x1 = torch.arange(torch.min(x_train[:,0]), torch.max(x_train[:,0]),5)
x2 = torch.arange(torch.min(x_train[:,1]), torch.max(x_train[:, 1]),5)

X1,X2 = torch.meshgrid(x1,x2)

#y = days
y = model.f(torch.cat((torch.ravel(X1).unsqueeze(1), torch.ravel(X2).unsqueeze(1)), dim=-1).type(torch.FloatTensor))
Y = y.reshape(X1.shape).detach()

ax.plot_wireframe(X1,X2,Y)
plt.show()
