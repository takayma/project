from library import *

class St:
	def __init__ (self, size, act):
		self.size = size
		act = eval(act + '()')
		self.f = act.f
		self.df = act.df