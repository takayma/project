from AI import *

matrix =   [[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]]

model = M()

dirr = os.getcwd() + '\\Images\\Normal\\'

os.chdir(dirr)
images = os.listdir()

for i in range(len(images)):
	images[i] = Image.open(dirr + images[i])
	images[i] = np.array(images[i])

image = images[0]

image = model.convolution(image, matrix)
image = np.array(image, dtype = np.uint8)

image = Image.fromarray(image, 'RGB')
rand = random.randint(0, 1000000)
dirr = dirr[:-7]
dirr += f'{rand}.jpg'

image.save(dirr)

# k = 6
# mm = np.array([[[random.randint(0, 255) for l in range(3)] for j in range(k)] for i in range(k)])







# model = St([3, 1], 1, .5)

# x = [
# 	{'input': [1, 1, 1], 'output': [1]},
# 	{'input': [1, 1, 0], 'output': [1]},
# 	{'input': [1, 0, 0], 'output': [0]},
# 	#{'input': [0, 0, 0], 'output': [0]},
# 	{'input': [0, 0, 1], 'output': [0]},
# 	#{'input': [0, 1, 1], 'output': [0]},
# 	{'input': [1, 0, 1], 'output': [0]},
# 	{'input': [0, 1, 0], 'output': [0]}
# 	]


# def MSE (x, ai):
# 	error = 0
# 	for i in x:
# 		ai.iteration(i['input'])
# 		for j in range(len(i['output'])):
# 			error += (i['output'][j] - ai.x[-1][j]) ** 2
# 		error /= len(i['output'])
# 	error /= len(x)

# 	return error

# def full_train (ai, x, a):
# 	epochs = 0

# 	while MSE(x, ai) > a:
# 		epochs += 1

# 		print(MSE(x, ai))

# 		i = random.randint(0, len(x) - 1)

# 		ai.iteration(x[i]['input'])
# 		ai.train(x[i]['output'])

# 	return epochs


# epochs = full_train(model, x, 0.001)

# print(epochs)

# model.iteration(x[0]['input'])
# print(model.x[-1])


# model.iteration(x[1]['input'])
# print(model.x[-1])


# model.iteration(x[2]['input'])
# print(model.x[-1])