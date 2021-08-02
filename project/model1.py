from library import *

class Perceptron:
	def __init__ (self, k, n, m, activation, loss):
		activation = eval(f'{activation}()')
		self.f = activation.f
		self.df = activation.df
		loss = eval(f'{loss}()')
		self.l_f = loss.f
		self.l_df = loss.df
		self.k = k
		self.n = n
		self.m = m
		self.w = [[[np.random.uniform(0, 1) for l in range(k[i + 1])] for j in range(k[i] + 1)] for i in range(len(k) - 1)]
		self.nw = [[[0 for l in range(k[i + 1])] for j in range(k[i] + 1)] for i in range(len(k) - 1)]

	def forward (self, input):
		self.x = [[0 for j in range(self.k[i])] for i in range(len(self.k))]
		self.x[0] = [*input]
		for i in range(len(self.x) - 1):
			self.x[i].append(1)

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.x[i + 1][j] += self.x[i][l] * self.w[i][l][j]
				if i != len(self.k) - 2:
					self.x[i + 1][j] = self.f(self.x[i + 1][j])
		self.x[-1] = Soft_Max().f(self.x[-1])

	def backward (self, output):
		self.err = [[0 for j in range(self.k[i])] for i in range(len(self.k))]

		for i in range(len(output)):
			self.err[-1][i] = self.l_df(output[i], self.x[-1][i])
		df = Soft_Max().df(self.x[-1])
		self.err[-1] = np.dot(self.err[-1], df)

		for i in range(len(self.k) - 2, -1):
			for j in range(self.k[i]):
				for l in range(self.k[i + 1]):
					self.err[i][j] += (self.err[i + 1][l] * self.w[i][j][l]) * self.df(self.x[i][j])

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.nw[i][l][j] = self.n * self.x[i][l] * self.err[i + 1][j] + self.m * self.nw[i][l][j]
					self.w[i][l][j] -= self.nw[i][l][j]

	def full_train (self, data):
		epochs = 0
		e = 1
		accuracy = 99
		while e > (100 - accuracy) / 100:
			epochs += 1

			i = np.random.randint(0, len(data))

			self.forward(data[i]['input'])
			self.backward(data[i]['output'])
			
			e = 0
			for element in data:
				inp = element['input']
				ide = element['output']
				self.forward(inp)
				out = self.x[-1]
				for ideal, output in zip(ide, out):
					e += f(ideal, output)
				e /= len(out)
			e /= len(data)
		return epochs