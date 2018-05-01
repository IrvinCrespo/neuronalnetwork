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

		res = np.sum((np.array(self.wh)*np.array(inp).T),axis=self.hidden-1)

		if len(res)>1:
			bias1 = []
			for b1 in res:
				bias1.append(random.uniform(-1,1))
			res+bias1
		else:
			res+=[random.uniform(-1,1),random.uniform(-1,1)]

		res = self.sigmoid(res)
		out = np.array(self.wo)*res.T

		if len(out) > 1:
			out = np.sum(out,axis=self.hidden-self.output)
			bias = []
			for b in out:
				bias.append(random.uniform(-1,1))
			out+bias
		else:
			out = np.sum(out[0],axis=0)
			out+[random.uniform(-1,1)]

		
		out = self.sigmoid(out)
		return out

	def f(self,x):
		return 1/(1+exp(-x))

	def sigmoid(self,mat):
		sig = np.vectorize(self.f)
		return sig(mat)



