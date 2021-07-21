
x = [
	#{'input': [1, 1, 1], 'output': [1]},
	{'input': [1, 1, 0], 'output': [1]},
	{'input': [1, 0, 0], 'output': [0]},
	#{'input': [0, 0, 0], 'output': [0]},
	{'input': [0, 0, 1], 'output': [0]},
	#{'input': [0, 1, 1], 'output': [0]},
	{'input': [1, 0, 1], 'output': [0]},
	{'input': [0, 1, 0], 'output': [0]}
	]

def full_train (ai, x, n, m, a):
	e = 1

	epochs = 0
	while e > a:
		epochs += 1

		i = random.randint(0, len(x) - 1)
		ai.iteration(x[i]['input'])
		ai.train(x[i]['output'], n, m)

		e = 0
		for j in x:
			ai.iteration(j['input'])
			for l in range(len(j['output'])):
				e += (j['output'][l] - ai.x[-1][l]) ** 2
			e /= len(j['output'])
		e /= len(x)

	return epochs

















	class St (Sigmoid):
	def __init__ (self, k, n, m):
		self.k = k
		self.n = n
		self.m = m
		self.w = [[[random.uniform(0, 1) for l in range(k[i + 1])] for j in range(k[i] + 1)] for i in range(len(k) - 1)]
		self.nw = [[[0 for l in range(k[i + 1])] for j in range(k[i] + 1)] for i in range(len(k) - 1)]

	def iteration (self, input):
		self.x = [[0 for j in range(self.k[i])] for i in range(len(self.k))]
		self.x[0] = input + [1]
		for i in range(1, len(self.x) - 1):
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
					self.nw[i][l][j] = n * self.x[i][l] * self.err[i + 1][j] + self.nw[i][l][j] * m
					self.w[i][l][j] += self.nw[i][l][j]