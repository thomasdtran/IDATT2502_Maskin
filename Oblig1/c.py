"""
Ikke-lineær regresjon i 2 dimensjoner (se neste side):
Lag en ikke-lineær modell som predikerer hodeomkrets ut fra alder (i dager) gitt
observasjonene i day_head_circumference.csv
Bruk følgende modell prediktor: f (x) = 20σ(xW + b) + 31, der σ er sigmoid
funksjonen som definert på neste slide.
"""


import torch
import matplotlib.pyplot as plt
import numpy as np

text = open("Oblig1/day_head_circumference.txt", "r")
text.readline()

day = [] 
circumference = []  
for line in text:
    l = line.split(",")
    day.append(float(l[0]))
    circumference.append(float(l[1]))

text.close()

n = len(day)  # number of observations
nt = 10  # number of tests

# Observed/training input and output
# x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
x_train = torch.tensor(day[0:n-nt]).reshape(-1, 1)
# y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]
y_train = torch.tensor(circumference[0:n-nt]).reshape(-1, 1)

#Data used for testing
x_test = torch.tensor(day[n-nt:n]).reshape(-1, 1)
y_test = torch.tensor(circumference[n-nt:n]).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        self.m = torch.nn.Sigmoid()

    # Predictor
    def f(self, x):
        # @ corresponds to matrix multiplication
        return 20 * self.m(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
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

#Use some of the testing data, and see how well the model performs
print("Testing")
print("--------")
for i in range(len(x_test)):
    print("Days = %s , Calculated head circumference = %s, Observed head circumference = %s" %
          (x_test[i], model.f(torch.tensor([[x_test[i]]])), y_test[i]))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Days')
plt.ylabel('Head circumference')

x = torch.range(torch.min(x_train), torch.max(x_train)).reshape(-1,1)

plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = 20σ(xW+b)+31$')
plt.legend()
plt.show()
