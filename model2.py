from library import *

def convolution (image, matrix, activation, stride):
	f = eval(activation)()
	f = f.f
	new_image = []
	image = [[0 for i in image[0]], *image, [0 for i in image[0]]]
	for i in range(len(image)):
		image[i] = [0, *image[i], 0]
	k = len(matrix)
	i = 0
	while i <= len(image) - k:
		new_image.append([])
		j = 0
		while j <= len(image[0]) - k:
			pic = np.array([image[i + l][j: j + k] for l in range(k)])
			pic = pic * matrix
			pic = sum(sum(pic))
			pic = f(pic)
			new_image[i].append(pic)
			j += stride
		i += stride

	return new_image

def max_pooling (image, size):
	new_image = []
	indexes = []
	i = 0
	while i <= len(image) - size:
		new_image.append([])
		indexes.append([])
		j = 0
		while j <= len(image[0]) - size:
			pic = np.array([image[i + l][j: j + size] for l in range(size)])
			index = (0, 0)
			for l in range(len(pic)):
				for m in range(len(pic[l])):
					if pic[index] < pic[l][m]: index = (l, m)
			new_image[round(i / size)].append(pic[index])
			indexes[round(i / size)].append((index[0] + i, index[1] + j))
			j += size
		i += size

	return new_image, indexes

def layers (layers):
	x = layer[0]['image']
	indexes = []
	for layer in layers:
		if layer['name'] == 'Conv':
			x = convolution(
				image = x,
				matrix = layer['matrix'],
				activation = layer['activation'],
				stride = layer['stride']
				)
		elif layer['name'] == 'Max_Pooling':
			x, index = max_pooling(
				image = x,
				size = layer['size']
				)
			indexes.append(index)