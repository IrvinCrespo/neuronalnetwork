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
		self.bias_h = np.random.uniform(low=-1, high=1, size=(hidden,1))
		self.bias_o = np.random.uniform(low=-1, high=1, size=(output,1))
		self.lrate = 0.1

	def feedForward(self,inp,ans):

	
		hidden = self.wh*np.matrix(inp).T
		hidden+=self.bias_h
		hidden = self.sigmoid(hidden)

		out = self.wo*hidden
		out +=self.bias_o
		out = self.sigmoid(out)

		if out>1:
			errors = self.subError(ans,out)
		else:
			errors = ans-out


		hidden_errors = self.wo.T*errors

		#calculate gradient
		gradient_o = self.dsigmoid(out)
		
		gradient_o = gradient_o*errors
		
		

		if isinstance(gradient_o, list):
			gradient_o = self.lr(gradient_o,self.lrate)
		else:
			gradient_o = gradient_o*self.lrate

		self.bias_o += gradient_o

		#calculate deltas, adding deltas to HO weights
	
		who_deltas = gradient_o*hidden.T
		self.wo += who_deltas
		

		#calculate hidden errors
		hg = self.dsigmoid(hidden)
		#print(hg)
		#print(hidden_errors)
		hg = hg*hidden_errors.T
		
		hg = self.lrate*hg.T#np.matrix(self.lrate).T#self.lr(hg,self.lrate)

	
		self.bias_h + hg
		wih_deltas = np.matrix(inp)*hg
		self.wh += wih_deltas
		#print(out)
		return out

	def f(self,x):
		return 1/(1+exp(-x))

	def sigmoid(self,mat):
		sig = np.vectorize(self.f)
		return sig(mat)

	def gradient(self,x):
		return x*(1-x)

	def dsigmoid(self,x):
		gra = np.vectorize(self.gradient)
		return gra(x)

	def multi(self,x,n):
		return x*n

	def multiplyEach(self,x,n):
		mu = np.vectorize(self.multi)
		return multi(x,n)

	def lr(self,a,rate):
		out = []
		for x in a:
			out.append(x*rate)
		return out

	def subError(self,a,b):
		error = []
		for x,y in zip(a,b):
			error.append(x-y)
		return error









		


