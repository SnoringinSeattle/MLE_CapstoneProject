#!/usr/bin/python

import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
def sigmoid_derivative(x):
	return x * (1 - x)

np.random.seed(324)

hidden_size = 4
training_steps = 100000

alphas = [0.01, 0.1, 1, 10, 100, 1000]

X = np.random.randint(2, size=(4, 3))
y = np.random.randint(2, size=(4, 1))
#print X
#print y

# Initialize 1st set of weights
W1 = np.random.rand(X.shape[1], hidden_size)

# Initialize 2nd set of weights
W2 = np.random.rand(hidden_size, y.shape[1])

for alpha in alphas:
	for i in range(training_steps):
		# Initialize hidden (fully connected) layer
		layer_1 = sigmoid(np.dot(X, W1))

		# Initialize y (fully connected) layer
		layer_2 = sigmoid(np.dot(layer_1, W2))

		# Get loss (MSE)
		layer_2_loss = y - layer_2
		
		if i % (training_steps/10) == 0: print alpha, "Error:" + str(np.mean(np.abs(layer_2_loss)))
			
		# Apply SGD to the loss: the more certain the estimate, the less weighted it will get: the gradient at the extremes is smaller than in the middle
		layer_2_wloss = layer_2_loss * sigmoid_derivative(layer_2) # element-wise multiplication!

		#print layer_2
		#print sigmoid_derivative(layer_2) 

		# Compute the effect of the hidden layer to the weighted loss
		layer_1_loss = np.dot(layer_2_wloss, W2.T)

		# Apply SGD
		layer_1_wloss = layer_1_loss * sigmoid_derivative(layer_1)

		# Update the weights
		W2 += alpha * np.dot(layer_1.T, layer_2_wloss)
		W1 += alpha * np.dot(X.T, layer_1_wloss)
