"""
Lag en modell med prediktoren f (x) = softmax(xW + b) som
klassifiserer handskrevne tall. Se mnist for eksempel lasting av
MNIST datasettet, og visning og lagring av en observasjon. Du
skal oppnå en nøyaktighet på 0.9 eller over. Lag 10 .png bilder
som viser W etter optimalisering.
"""
import torch
import matplotlib.pyplot as plt
import torchvision

class SoftmaxModel:
    def __init__(self):
        # Model variables
        self.W = torch.randn(784, 10, requires_grad=True)
        self.b = torch.randn(1, 10, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))
    # Similar to:
    # return -torch.mean(y * torch.log(self.f(x)) +
    # (1 - y) * torch.log(1 - self.f(x)))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output


model = SoftmaxModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=1)
for epoch in range(300):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

    if((model.accuracy(x_test, y_test)) >= 0.91): #Stop when we have wanted accuracy
        break

print("Accuray = {}".format(model.accuracy(x_test, y_test)))


#Creates 10 .png of W
for i in range(10):
    plt.imsave('oblig2/w_images/w_{}.png'.format(i+1), model.W[:,i].reshape(28, 28).detach().numpy())

#Plots all the images for W
plt.figure(figsize=(20,2)) #Gives the window a width of 20 and length 2
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.tight_layout()
    plt.imshow(model.W[:, i].reshape(28, 28).detach().numpy()) #Each column represents each image.
    plt.title("W{}".format(i+1))
    plt.xticks([])
    plt.yticks([])

plt.show()

