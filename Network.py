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
		
		i = np.matrix(inp)

		x1 = np.array(self.wh)
		x2 = np.array(inp)

		res = (x1*x2.T)
		res = np.sum(res,axis=1)
		res+=[random.uniform(-1,1),random.uniform(-1,1)]
		res = self.sigmoid(res)

		out = np.array(self.wo)*res.T
		out = np.sum(out[0],axis=0)
		print(out)
		out+[random.uniform(-1,1),random.uniform(-1,1)]
		out = self.sigmoid(out)
		print(out)
		return out

	def f(self,x):
		return 1/(1+exp(-x))

	def sigmoid(self,mat):
		sig = np.vectorize(self.f)
		return sig(mat)



