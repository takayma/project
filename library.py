import os
import numpy as np
from random import *
from PIL import Image
from pprint import pprint
from math import *
import json
from copy import deepcopy

# output == y
# ideal == y_hat


class MSE:
	def f(self, y, y_hat):
		return (y - y_hat) ** 2

	def df(self, y, y_hat):
		return y - y_hat


class MAE:
	def f(self, y, y_hat):
		return abs(y - y_hat)

	def df(self, y, y_hat):
		if y > y_hat:
			return 1
		elif y_hat < y:
			return -1
		else:
			return 0


class Huber_Loss:
	def __init__(self, delta):
		self.delta = delta

	def f(self, y, y_hat):
		if abs(y - y_hat) < self.delta:
			return 1/2 * (y - y_hat) ** 2
		else:
			return self.delta * (y - y_hat - 1/2 * self.delta)

class Binary_Cross_Entropy:
	def f(self, y, y_hat):
		return -(y_hat * log(y) + (1 - y_hat) * log(1 - y))

	def df(self, y, y_hat):
		return -(y_hat / y) + (1 - y_hat) / (1 - y)


class Sigmoid:
	def f(self, x):
		return 1 / (1 + exp(-x))

	def df(self, x):
		return (1 - x) * x


class Tangh:
	def f(self, x):
		return (exp(2 * x) - 1) / (exp(2 * x) + 1)

	def df(self, x):
		return 1 - x ** 2


class ReLU:
	def f(self, x):
		return max(x, 0)

	def df(self, x):
		if x > 0:
			return 1
		else:
			return 0


class Soft_Max:
	def f(self, x):
		x = [exp(i) for i in x]
		s = sum(x)
		x = [i / s for i in x]
		return x

	def df(self, x):
		return x * (1 - x)
