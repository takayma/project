import random
import math
import os
from PIL import Image
import numpy as np

class Sigmoid:
	def f (self, x):
		return 1 / (1 + math.exp(-x))

	def df (self, x):
		return (1 - x) * x


class Th:
	def f (self, x):
		return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

	def df (self, x):
		return 1 - self.f(x) ** 2


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