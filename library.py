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
		returnmax(x, 0.01 * x)

	def df (self, x):
		if x >= 0:
			return 1
		else:
			return 0.01