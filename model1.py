from library import *

class St (Sigmoid):
	def __init__ (self, k, n):
		super().__init__()
		self.k = k
		self.n = n
		self.w = [[[random.uniform(0, 1) for l in range(k[i + 1])] for j in range(k[i] + 1)] for i in range(len(k) - 1)]
		
	def iteration (self, input):
		self.x = [[0 for j in range(self.k[i])] for i in range(len(self.k))]
		self.x[0] = [*input]
		for i in range(len(self.x) - 1):
			self.x[i].append(1)

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.x[i + 1][j] += self.x[i][l] * self.w[i][l][j]
				self.x[i + 1][j] = self.f(self.x[i + 1][j])


	def train (self, output):
		self.err = [[0 for j in range(self.k[i])] for i in range(len(self.k))]

		for i in range(len(output)):
			self.err[-1][i] = (output[i] - self.x[-1][i]) * self.df(self.x[-1][i])

		for i in range(len(self.k) - 2, -1):
			for j in range(self.k[i]):
				for l in range(self.k[i + 1]):
					self.err[i][j] += (self.err[i + 1][l] * self.w[i][j][l]) * self.df(self.x[i][j])

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.w[i][l][j] += self.n * self.x[i][l] * self.err[i + 1][j]