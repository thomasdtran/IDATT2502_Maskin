"""
Lineær regresjon i 2 dimensjoner:
Lag en lineær modell som predikerer vekt ut fra lengde gitt observasjonene i
length_weight.csv
"""

import torch
import matplotlib.pyplot as plt

text = open("Oblig1/newborn.txt", "r")
text.readline()

length = [] #Length
weight = [] #Weight
for line in text:
    l = line.split(",")
    length.append(float(l[0]))
    weight.append(float(l[1]))

text.close()

n = len(length) #number of observations
nt = 10 #number of tests
# Observed/training input and output
x_train = torch.tensor(length[0:n-nt]).reshape(-1, 1)  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
y_train = torch.tensor(weight[0:n-nt]).reshape(-1, 1)  # y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]

#Data used for testing
x_test = torch.tensor(length[n-nt:n]).reshape(-1, 1)
y_test = torch.tensor(weight[n-nt:n]).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(400000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

#Use some of the testing data, and see how well the model performs
print("Testing")
print("--------")
for i in range(len(x_test)):
    print("Length = %s , Calculated weight = %s, Observed weight = %s" % (x_test[i], model.f(torch.tensor([[x_test[i]]])), y_test[i]))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]]) 
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()


