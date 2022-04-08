from library import *

relu = ReLU()
softmax = Soft_Max()

a = [2, 1, 0.1]

for i in range(len(a)):
    a[i] = relu.f(a[i])

print(a)

a = softmax.f(a)

print(a)