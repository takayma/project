from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.graphics import (Color, Line)

from library import *
from model import *


ai = NN(ls=[['full_connected', 800*600, 20, 2]])

class paint(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(255, 255, 255)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 2)

    def on_touch_move(self, touch):
        touch.ud['line'].points += (touch.x, touch.y)

class main(App):
    def build(self):
        self.lb = Label(size=(100, 50), text='OK', pos=(700, 0))
        parent = Widget()
        self.painter = paint()
        parent.add_widget(self.painter)
        parent.add_widget(Button(size=(100, 50), text='Clear', on_press=self.clear_canvas))
        parent.add_widget(Button(size=(100, 50), text='Save as 1', on_press=self.save_as_1, pos=(100, 0)))
        parent.add_widget(Button(size=(100, 50), text='Save as 0', on_press=self.save_as_0, pos=(200, 0)))
        parent.add_widget(Button(size=(100, 50), text='Train', on_press=self.train_nn, pos=(300, 0)))
        parent.add_widget(Button(size=(100, 50), text='Save weights', on_press=self.save_weights, pos=(400, 0)))
        parent.add_widget(Button(size=(100, 50), text='Load weights', on_press=self.load_weights, pos=(500, 0)))
        parent.add_widget(Button(size=(100, 50), text='Check', on_press=self.train_nn, pos=(600, 0)))
        parent.add_widget(self.lb)

        return parent

    def clear_canvas (self, instance):
        self.painter.canvas.clear()

    def save_as_1 (self, instance):
        self.painter.size = (Window.size[0], Window.size[1])
        arr = os.listdir('C:\\Users\\takayma\\Desktop\\project\\1')
        x = 0
        if len(arr) != 0:
            x = int(arr[-1][:-4]) + 1
        self.painter.export_to_png(f'C:\\Users\\takayma\\Desktop\\project\\1\\{x}.png')

    def save_as_0 (self, instance):
        self.painter.size = (Window.size[0], Window.size[1])
        arr = os.listdir('C:\\Users\\takayma\\Desktop\\project\\0')
        x = 0
        if len(arr) != 0:
            x = int(arr[-1][:-4]) + 1
        self.painter.export_to_png(f'C:\\Users\\takayma\\Desktop\\project\\0\\{x}.png')

    def train_nn (self, instance):
        os.chdir('C:\\Users\\takayma\\Desktop\\project\\1')
        train_set = []
        for i in os.listdir():
            dirr = os.getcwd() + '\\' + i
            image = Image.open(dirr)
            image = np.array(image)
            arr = []
            for y in range(len(image)):
                for x in range(len(image[0])):
                    arr.append(all(image[y][x] == [255,255,255,255]))
            train_set.append([arr, [1, 0]])
        os.chdir('C:\\Users\\takayma\\Desktop\\project\\0')
        for i in os.listdir():
            dirr = os.getcwd() + '\\' + i
            image = Image.open(dirr)
            image = np.array(image)
            arr = []
            for y in range(len(image)):
                for x in range(len(image[0])):
                    arr.append(all(image[y][x] == [255,255,255,255]))
            train_set.append([arr, [0, 1]])

        ai.train_fc(train_set, 0.01)
        print('complete')

    def save_weights(self, instance):
        self.lb.text = 'Saving...'
        with open('weights.txt', 'w') as fw:
            json.dump(ai.w, fw)
        self.lb.text = 'Saving complete'

    def load_weights(self, instance):
        self.lb.text = 'Loading...'
        with open('weights.txt', 'r') as fr:
            ai.w = json.load(fr)
        self.lb.text = 'Loading complete'


main().run()