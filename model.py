from library import *


class NN:
	def __init__(self, ls, activation='ReLU', loss='Cross_Entropy', softmax=True, n=0.4, m=0.8):
		self.ls = ls

		activation = eval(f'{activation}()')
		self.f = activation.f
		self.df = activation.df

		loss = eval(f'{loss}()')
		self.l_f = loss.f
		self.l_df = loss.df

		self.softmax = softmax

		self.matrixes = []
		for layer in ls:
			if layer[0] == 'convolution':
				self.matrixes.append([])
				for i in range(layer[1]):
					self.matrixes[-1].append(
						[[uniform(1, 5) for _1 in range(layer[2])] for _2 in range(layer[3])])
			elif layer[0] == 'full_connected':
				self.k = layer[1:]

		self.n = n
		self.m = m

		self.w = []
		self.delta_w = []
		for i in range(len(self.k) - 1):
			self.w.append([])
			self.delta_w.append([])
			for j in range(self.k[i] + 1):
				self.w[i].append([])
				self.delta_w[i].append([])
				for l in range(self.k[i + 1]):
					self.w[i][j].append(uniform(0, 1))
					self.delta_w[i][j].append(0)

	def __copy__(self):
		copied = NN(ls=self.ls, softmax=self.softmax, n=self.n, m=self.m)
		copied.f = self.f
		copied.df = self.df
		copied.l_f = self.l_f
		copied.l_df = self.l_df

		return copied

	def forward_fc(self, input):
		self.x = [[0 for j in range(self.k[i])] for i in range(len(self.k))]
		self.x[0] = [*input]
		for i in range(len(self.x) - 1):
			self.x[i].append(1)

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.x[i + 1][j] += self.x[i][l] * self.w[i][l][j]
				if i != len(self.k) - 2 or not self.softmax:
					self.x[i + 1][j] = self.f(self.x[i + 1][j])
		if self.softmax:
			self.x[-1] = Soft_Max().f(self.x[-1])

	def backpropagation_fc(self, output):
		self.err = [[0 for j in range(self.k[i])] for i in range(len(self.k))]

		for i in range(len(output)):
			self.err[-1][i] = self.l_df(output[i], self.x[-1][i])
			if self.softmax:
				self.err[-1][i] *= Soft_Max().df(self.x[-1][i])
			else:
				self.err[-1][i] *= self.df(self.x[-1][i])

		for i in range(len(self.k) - 2, -1):
			for j in range(self.k[i]):
				for l in range(self.k[i + 1]):
					self.err[i][j] += (self.err[i + 1][l] * self.w[i][j][l]) * self.df(self.x[i][j])

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for l in range(self.k[i] + 1):
					self.delta_w[i][l][j] = self.m * self.delta_w[i][l][j] + self.n * self.x[i][l] * self.err[i + 1][j]
					self.w[i][l][j] -= self.delta_w[i][l][j]

	def train_fc(self, data, a=0.001):
		epochs = 0
		e = a + 1
		while e > a:
			ind = randint(0, len(data) - 1)
			self.forward_fc(data[ind][0])
			self.backpropagation_fc(data[ind][1])

			e = 0
			for input, output in data:
				self.forward_fc(input)
				for i in range(len(output)):
					e += self.l_f(output[i], self.x[-1][i])
				e /= len(output)
			e /= len(data)
			print(e)

			epochs += 1
		return epochs

	def genetic_algorithm(self, inp, start_count, best_count):
		arr = []
		for i in range(start_count):
			arr.append(self.__copy__())
			arr[i].forward_fc(inp)
			arr[i].calculate_error()

		arr.sort(key=lambda x: x.err)
