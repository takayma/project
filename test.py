from model import *

a = NN(layers=[['full_connected', 3, 2]])

x = [
	[[1, 1, 1], [1, 0]],
	[[1, 1, 0], [1, 0]],
	[[1, 0, 0], [0, 1]],
	[[0, 0, 0], [0, 1]],
	[[0, 0, 1], [0, 1]],
	[[0, 1, 1], [0, 1]],
	[[1, 0, 1], [0, 1]],
	[[0, 1, 0], [0, 1]]
	]

a.train_fc(x)

while 1:
	inp = input('Input::: ')
	inp = list(map(int, list(inp)))
	a.forward_fc(inp)
	print(a.x[-1])