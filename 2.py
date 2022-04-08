from model2 import *

x = [[[randint(0, 255) for l in range(3)] for j in range(6)] for i in range(6)]

print(x)

ls =   [['convolution', 1, 3, 3],
		['max_pooling', 2, 2],
		['max_pooling', 3, 3],
		['full_connected', 1, 2, 1]]

conv = NN(
	ls=ls,
	softmax=False
)

def from_rgb(image):
	r, g, b = [], [], []
	for layer in image:
		r.append([])
		g.append([])
		b.append([])
		for r0, g0, b0 in layer:
			r[-1].append(r0)
			g[-1].append(g0)
			b[-1].append(b0)
	return r, g, b

r, g, b = from_rgb(x)

arr = conv.forward_conv(r)
print(arr)
print(conv.x)
# dirr = r'C:\Users\takayma\Desktop\project\Images\1.png'
# image = Image.open(dirr)
# image = np.array(image)