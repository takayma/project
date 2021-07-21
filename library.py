import random
import math

class Sigmoid:
	def f (self, x):
		return 1 / (1 + math.exp(-x))

	def df (self, x):
		return (1 - x) * x

class ReLU:
	def f (self, x):
		if x >= 0:
			return x
		else:
			return 0

	def df (self, x):
		if x >= 0:
			return 1
		else:
			return 0

class Leaky_ReLU:
	def f (self, x):
		if x >= 0:
			return x
		else:
			return x * 0.01

	def df (self, x):
		if x >= 0:
			return 1
		else:
			return 0.01