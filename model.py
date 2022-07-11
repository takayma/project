from library import *


class NN:
	def __init__(self, layers, activation=Sigmoid, loss=Binary_Cross_Entropy, softmax=True, n=0.4, m=0.8):
		self.error = None
		self.x = None
		self.err = None

		self.layers = layers

		self.activation = activation
		self.f = activation().f
		self.df = activation().df

		self.loss = loss
		self.l_f = loss().f
		self.l_df = loss().df

		self.softmax = softmax

		self.matrices = []
		for layer in layers:
			if layer[0] == 'convolution':
				self.matrices.append([])
				for i in range(layer[1]):
					self.matrices[-1].append([[uniform(1, 5) for _1 in range(layer[2])] for _2 in range(layer[3])])
			elif layer[0] == 'full_connected':
				self.k = layer[1:]

		self.n = n
		self.m = m

		self.w = [[[uniform(0, 1) for _1 in range(self.k[i + 1])] for _2 in range(self.k[i] + 1)] for i in range(len(self.k) - 1)]
		self.delta_w = [[[0 for _1 in range(self.k[i + 1])] for _2 in range(self.k[i] + 1)] for i in range(len(self.k) - 1)]

	def __copy__(self):
		copied = NN(layers=self.layers, softmax=self.softmax, n=self.n, m=self.m)
		copied.f = self.f
		copied.df = self.df
		copied.l_f = self.l_f
		copied.l_df = self.l_df

		return copied

	def forward_fc(self, data_input):
		self.x = [[0 for _1 in range(self.k[i])] for i in range(len(self.k))]
		self.x[0] = [*data_input]
		for i in range(len(self.x) - 1):
			self.x[i].append(1)

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for d in range(self.k[i] + 1):
					self.x[i + 1][j] += self.x[i][d] * self.w[i][d][j]
				if i != len(self.k) - 2 or not self.softmax:
					self.x[i + 1][j] = self.f(self.x[i + 1][j])
		if self.softmax:
			self.x[-1] = Soft_Max().f(self.x[-1])

	def backpropagation_fc(self, data_output):
		self.err = [[0 for _1 in range(self.k[i])] for i in range(len(self.k))]

		for i in range(len(data_output)):
			self.err[-1][i] = self.l_df(self.x[-1][i], data_output[i])
			if self.softmax:
				self.err[-1][i] *= Soft_Max().df(self.x[-1][i])
			else:
				self.err[-1][i] *= self.df(self.x[-1][i])

		for i in range(len(self.k) - 2, -1):
			for j in range(self.k[i]):
				for d in range(self.k[i + 1]):
					self.err[i][j] += self.err[i + 1][d] * self.w[i][j][d] * self.df(self.x[i][j])

		for i in range(len(self.k) - 1):
			for j in range(self.k[i + 1]):
				for d in range(self.k[i] + 1):
					self.delta_w[i][d][j] = self.m * self.delta_w[i][d][j] + self.n * self.x[i][d] * self.err[i + 1][j]
					self.w[i][d][j] -= self.delta_w[i][d][j]

	def calculate_error(self, data):
		self.error = 0
		for data_input, data_output in data:
			self.forward_fc(data_input)
			for ideal, output in zip(data_output, self.x[-1]):
				self.error += self.l_f(output, ideal)
				if self.loss == MAE or self.loss == MSE:
					self.error /= len(data_output)
		self.error /= len(data)

	def genetic_algorithm(self, data, count, best_count, arr=[]):
		if arr == []:
			for i in range(count):
				arr.append(self.__copy__())
		else:
			old = deepcopy(arr)
			arr = []
			for i1 in range(count):
				arr.append(self.__copy__())
				for i in range(len(self.k) - 1):
					for j in range(self.k[i + 1]):
						for d in range(self.k[i] + 1):
							index = randint(0, best_count - 1)
							arr[i1].w[i][d][j] = old[index].w[i][d][j] + uniform(-0.5, 0.5)

		for i in range(count):
			arr[i].forward_fc(data_input)
			arr[i].calculate_error(data)

		arr.sort(key=lambda x: x.error)

		for a in arr[: best_count - 1]:
			print(a.error)

		return arr[: best_count]

	def train_fc(self, data, a=0.001):
		epochs = 0
		while True:
			ind = randint(0, len(data) - 1)
			self.forward_fc(data[ind][0])
			self.backpropagation_fc(data[ind][1])

			self.calculate_error(data)
			print(self.error)

			epochs += 1

			if self.error < a:
				break

		print(epochs)
