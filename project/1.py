from tkinter import *
from AI import *


root = Tk()

width = 200
height = 200

canv = Canvas(root, width = width, height = height, background = 'white')
canv.pack()

model = Perceptron(
        k = [2, 2],
        n = 1,
        m = .5,
        activation = 'Sigmoid',
        loss = 'Cross_Entropy'
        )

data = []

r = 5

def dot (event):
    for x in range(width):
        for y in range(height):
            x0, y0 = x / width, y / height
            model.forward([x0, y0])
            result = model.x[-1][0]
            if result >= 0.85:
                color = 'red'
            else:
                color = 'green'
            r = 0.5
            canv.create_oval(x - r, y - r, x + r, y + r, fill = color)
            

def red_dot (event):
    x, y = event.x, event.y
    canv.create_oval(x - r, y - r, x + r, y + r, fill = 'red')
    x, y = x / width, y / height
    data.append({'input': [x, y], 'output': [1, 0]})
    model.full_train(data)

def green_dot (event):
    x, y = event.x, event.y
    canv.create_oval(x - r, y - r, x + r, y + r, fill = 'green')
    x, y = x / width, y / height
    data.append({'input': [x, y], 'output': [0, 1]})
    model.full_train(data)

root.bind('z', red_dot)
root.bind('x', green_dot)
root.bind('c', dot)
root.mainloop()