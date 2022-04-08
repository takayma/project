from library import *


class NN:
	def __init__(self, ls, activation = 'ReLU', loss = 'Cross_Entropy', softmax = True, a = 0.4, b = 0.8):
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
		self.a = a
		self.b = b
		self.w = [[[uniform(0, 1) for l in range(self.k[i + 1])] for j in range(self.k[i] + 1)] for i in range(len(self.k) - 1)]
		self.delta_w = [[[0 for l in range(self.k[i + 1])] for j in range(self.k[i] + 1)] for i in range(len(self.k) - 1)]

	def forward_fc (self, input):
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

	def backward_fc(self, output):
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
					self.delta_w[i][l][j] = self.b * self.delta_w[i][l][j] + self.a * self.x[i][l] * self.err[i + 1][j]
					self.w[i][l][j] -= self.delta_w[i][l][j]

	def train_fc(self, data, a = 0.001):
		epochs = 0
		e = a + 1
		while e > a:
			ind = randint(0, len(data) - 1)
			self.forward_fc(data[ind][0])
			self.backward_fc(data[ind][1])

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














	def convolution (self, image, matrix, stride_h = 1, stride_w = 1):
		new_image = []
		image = [[0 for i in range(len(image[0]))]] + image + [[0 for i in range(len(image[0]))]]
		for i in range(len(image)):
			image[i] = [0] + image[i] + [0]

		image_h = len(image)
		image_w = len(image[0])
		matrix_h = len(matrix)
		matrix_w = len(matrix[0])

		for y in range(0, image_h - matrix_h + 1, stride_h):
			new_image.append([])
			for x in range(0, image_w - matrix_w + 1, stride_w):
				s = 0
				for i in range(matrix_h):
					for j in range(matrix_w):
						s += image[y + i][x + j] * matrix[i][j]
				s = self.f(s)
				new_image[y].append(s)

		return new_image

	def max_pooling (self, image, stride_w = 2, stride_h = 2):
		new_image = []
		image_w = len(image[0])
		image_h = len(image)
		for y in range(0, image_h, stride_h):
			new_image.append([])
			for x in range(0, image_w, stride_w):
				s = image[y][x]
				for i in range(stride_h):
					for j in range(stride_w):
						s = max(s, image[y + i][x + j])
				new_image[int(y / 2)].append(s)

		return new_image

	def forward_conv (self, image):
		i = 0
		images = [image]
		for layer in self.ls:
			if layer[0] == 'convolution':
				new_images = []
				for image in images:
					for matrix in self.matrixes[i]:
						new_images.append(self.convolution(image, matrix))
				images = new_images
				i += 1
			elif layer[0] == 'max_pooling':
				for i in range(len(images)):
					images[i] = self.max_pooling(images[i], layer[1], layer[2])
			elif layer[0] == 'full_connected':
				new_image = []
				for image in images:
					for layer0 in image:
						new_image += layer0
				images = new_image
				self.forward_fc(images)
		return images