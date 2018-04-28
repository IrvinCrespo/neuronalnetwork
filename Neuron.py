from math import exp
class Neuron():
	u = 0.5
	t = 0.1
	b = 1

	"""docstring for Neuron"""
	def __init__(self,inputs=None,weigths=None):
		self.inputs = inputs
		self.weigths = weigths

	def sigmoid(self,x):
		return (1/(1+exp(-x)))

	def sumatory(self,entradas,pesos):
		suma = sum(i*p for i,p in zip(entradas,pesos))
		return suma
		#val = self.sigmoid(suma+self.b)
		#if val > 0:
		#	return 1 
		#else:
		#	return val



		