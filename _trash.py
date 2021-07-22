from AI import *

model = St([3, 1], 1, .5)

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


def MSE (x, ai):
	error = 0
	for i in x:
		ai.iteration(i['input'])
		for j in range(len(i['output'])):
			error += (i['output'][j] - ai.x[-1][j]) ** 2
		error /= len(i['output'])
	error /= len(x)

	return error

def full_train (ai, x, a):
	epochs = 0

	while MSE(x, ai) > a:
		epochs += 1

		print(MSE(x, ai))

		i = random.randint(0, len(x) - 1)
		ai.iteration(x[i]['input'])
		ai.train(x[i]['output'])

	return epochs


epochs = full_train(model, x, 0.001)

print(epochs)

matrix =   [[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]]

# Sigmoid: (4833 + 4778 + 4899 + 4847 + 4844) / 5 = 4840.2
# ReLU: () / 5 =
# Leaky_ReLU: () / 5 =
