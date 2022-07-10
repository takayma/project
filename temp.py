def convolution(self, image, matrix, stride_h=1, stride_w=1):
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


def max_pooling(self, image, stride_w=2, stride_h=2):
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


def forward_conv(self, image):
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