from AI import *

# matrix =   [[1, 0, 0],
# 			[0, 1, 0],
# 			[0, 0, 1]]

# dirr = os.getcwd() + '\\Images\\For Test\\'

# os.chdir(dirr)
# images = os.listdir()

# for i in range(len(images)):
# 	images[i] = Image.open(dirr + images[i])
# 	images[i] = np.array(images[i])

# 	image = images[i]

# 	image = convolution(image, matrix)
# 	image = np.array(image, dtype = np.uint8)

# 	image = Image.fromarray(image, 'RGB')
# 	rand = np.random.randint(0, 1000000)
# 	dirr = dirr[:-9]
# 	dirr += f'{rand}.jpg'

# 	image.save(dirr)

# k = 6
# test = np.array([[np.random.randint(0, 255) for j in range(k)] for i in range(k)])

# print(test)
# test = convolution(test, matrix)
# print(test)
# test, indexes = max_pooling(test, 2)
# print(test, indexes)



model = Perceptron(
		k = [3, 1],
		n = 1,
		m = .5,
		activation = 'Sigmoid'
		)

x = [
	{'input': [1, 1, 1], 'output': [1]},
	{'input': [1, 1, 0], 'output': [1]},
	{'input': [1, 0, 0], 'output': [0]},
	#{'input': [0, 0, 0], 'output': [0]},
	{'input': [0, 0, 1], 'output': [0]},
	#{'input': [0, 1, 1], 'output': [0]},
	{'input': [1, 0, 1], 'output': [0]},
	{'input': [0, 1, 0], 'output': [0]}
	]

def full_train (model, x, a):
	epochs = 0

	while MSE(model, x) > a:
		epochs += 1

		print(MSE(model, x))

		i = np.random.randint(0, len(x))

		model.forward(x[i]['input'])
		model.backward(x[i]['output'])

	return epochs

epochs = full_train(model, x, .001)

print(epochs)


while True:
	inp = input()
	inp = list(inp)
	inp = list(map(int, inp))
	model.forward(inp)
	print(model.x[-1])