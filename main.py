from AI import *

layers = [St(3, 'Leaky_ReLU')]
model = AI.create(layers)

print(model)


# model = AI.create([
# 	AI.St(size = 3),
# 	AI.St(size = 4),
# 	AI.St(size = 1)
# 	])