import os
from PIL import Image
import numpy as np
from random import *
from pprint import pprint
from math import *

class MSE:
	def f (self, ideal, output):
		return 1/2 * (ideal - output) ** 2
	def df (self, ideal, output):
		return output - ideal

class Cross_Entropy:
	def f (self, ideal, output):
		return -(ideal * log(output) + (1 - ideal) * log(1 - output))
	def df (self, ideal, output):
		return -(ideal / output) + (1 - ideal) / (1 - output)

class Sigmoid:
	def f (self, x):
		return 1 / (1 + exp(-x))

	def df (self, x):
		return (1 - x) * x

class Tangh:
	def f (self, x):
		return (exp(2 * x) - 1) / (exp(2 * x) + 1)

	def df (self, x):
		return 1 - x ** 2

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
		x = [exp(i) for i in x]
		s = sum(x)
		x = [i / s for i in x]
		return x

	def df (self, x):
		return x * (1 - x)