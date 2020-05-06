import numpy as np
import mnist

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn((n_inputs,n_neurons))
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        return np.dot(inputs, self.weights) + self.biases

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

#normalize pixel values from [0,255] to [-0.5,0.5]

train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

#Flatten the Images. Flatten the 28x28 images into a 784 vector

train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))



