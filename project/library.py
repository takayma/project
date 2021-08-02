import os
from PIL import Image
import numpy as np


def f (ideal, output):
	return abs(ideal - output)


class MSE:
	def f (self, ideal, output):
		return 1/2 * (ideal - output) ** 2

	def df (self, ideal, output):
		return -(ideal - output)

	
class Cross_Entropy:
	def f (self, ideal, output):
		return -(ideal * np.log(output))

	def df (self, ideal, output):
		return -(ideal / output)


class Sigmoid:
	def f (self, x):
		return 1 / (1 + np.exp(-x))

	def df (self, x):
		return (1 - x) * x


class ReLU:
	def f (self, x):
		return max(x, 0)

	def df (self, x):
		if x > 0:
			return 1
		else:
			return 0


class Soft_Max:
	def f (self, x):
		return np.exp(x) / sum(np.exp(x))

	def df (self, x):
		x = np.array(x)
		x = x.reshape(-1,1)
		return np.diagflat(x) - np.dot(x, x.T)