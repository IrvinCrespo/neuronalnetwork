from FNeuron import FNeuron
import random
import numpy as np
from math import exp

inputs = [[1,0],[0,1],[1,1]]

weigths = [0.1,0.2,0.1]

class Network():
	
	def __init__(self,inputs,hidden,output):
		self.inputs = inputs
		self.hidden = hidden
		self.output = output
		self.ex = np.random.uniform(low=-1, high=1, size=(hidden,inputs))
		self.wh = np.random.uniform(low=-1, high=1, size=(hidden,inputs))
		self.wo = np.random.uniform(low=-1, high=1, size=(output,hidden))


	def feedForward(self,inp):
		
		#res = self.wh * inp[:, 0]
		i = np.matrix(inp)
		#res = self.wh * i[:, None]
		print(i.T)

		x1 = np.array(self.wh)
		x2 = np.array(inp)
		print(x1)
		print(x2)
		res = (x1*x2)
		
		print(res)
		#res = np.multiply(self.wh*inp)
		print(res)
		res = self.ex * inp
		res = np.sum(res,axis=1)
		print(res)
		res+=[random.uniform(-1,1),random.uniform(-1,1)]
		print(res)
		res = self.sigmoid(res)
		print(res)

	def f(self,x):
		return 1/(1+exp(-x))

	def sigmoid(self,mat):
		sig = np.vectorize(self.f)
		return sig(mat)



