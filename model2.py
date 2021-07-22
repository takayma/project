from library import *

class M:
	def convolution (self, image, matrix, stride = 1):
		new_image = []
		image = [[[0, 0, 0] for i in image[0]], *image, [[0, 0, 0] for i in image[0]]]
		for i in range(len(image)):
			image[i] = [[0, 0, 0], *image[i], [0, 0, 0]]
		k = len(matrix)
		i = 0
		while i <= len(image) - k:
			new_image.append([])
			j = 0
			while j <= len(image[0]) - k:
				rgb = np.array([image[i + l][j: j + k] for l in range(k)])
				r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
				r, g, b = np.array([r, g, b])  * matrix
				r, g, b = sum(sum(r)), sum(sum(g)), sum(sum(b))
				r, g, b = round(r / k), round(g / k), round(b / k)
				rgb = [r, g, b]
				new_image[i].append(rgb)
				j += stride
			i += stride

		return new_image

	def max_pooling (sef, image, size):
		new_image = []
		i = 0
		while i <= len(image) - size:
			new_image.append([])
			j = 0
			while j <= len(image[0]) - size:
				rgb = np.array([image[i + l][j: j + size] for l in range(size)])
				r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
				nr, ng, nb = [], [], []
				for l in range(len(r)):
					for m in range(len(r[l])):
						nr.append(r[l][m])
						ng.append(g[l][m])
						nb.append(b[l][m])
				r, g, b = max(nr), max(ng), max(nb)
				rgb = [r, g, b]
				new_image[round(i / size)].append(rgb)
				j += size
			i += size

		return new_image