import os
from PIL import Image
import numpy as np


def MSE (ai, x):
	error = 0
	for i in x:
		ai.forward(i['input'])
		for j in range(len(i['output'])):
			error += (i['output'][j] - ai.x[-1][j]) ** 2
		error /= len(i['output'])
	error /= len(x)

	return error
	
def Cross_Entropy (ai, x):
	error = 0
	for i in x:
		ai.forward(i['input'])
		for j in range(len(i['output'])):
			error += -(np.log(ai.x[-1][j]) * i['output'][j])
		error /= len(i['output'])
	error /= len(x)

	return error


class Sigmoid:
	def f (self, x):
		return 1 / (1 + np.exp(-x))

	def df (self, x):
		return (1 - x) * x


class ReLU:
	def f (self, x):
		return max(x, 0)

	def df (self, x):
		if x >= 0:
			return 1
		else:
			return 0


class Leaky_ReLU:
	def f (self, x):
		return max(x, 0.01 * x)

	def df (self, x):
		if x >= 0:
			return 1
		else:
			return 0.01