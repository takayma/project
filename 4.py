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
	layers=[['full_connected', 3, 2, 2]]
	)

a = []

for i in range(100):
	for j in x:
		a = model.genetic_algorithm(j[0], x, 20, 2, a)

model = a[0]

for test in x:
	model.forward_fc(test[0])
	model.calculate_error(x)
	print(f'Input = {test[0]}, ideal = {test[1]}, output = {model.x[-1]}, error = {model.error}')