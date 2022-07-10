from model import *

x = [
	[[1, 1, 1], [1, 0]],
	[[1, 1, 0], [1, 0]],
	[[1, 0, 0], [0, 1]],
	[[0, 0, 1], [0, 1]],
	[[1, 0, 1], [0, 1]],
	[[0, 1, 0], [0, 1]]
	]

model = NN(
	ls=[['full_connected', 3, 2]],
	)
print(model.train_fc(x))

while 1:
	inp = list(map(int, list(input('Input::: '))))
	model.forward_fc(inp)
	print(model.x[-1])