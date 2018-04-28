from Neuron import Neuron
import random
import numpy as np
from math import exp

inp = [1,0]

inputs = [[1,0]]
targets = [1]
inputsx = [[1,0],[1,1],[0,1],[0,0]]

def f(x):
    return 1/(1+exp(-x))
    # return np.sqrt(x)



def feedForward(inputs):
	

	w11 = random.uniform(-1,1)
	w12 = random.uniform(-1,1)
	w21 = random.uniform(-1,1)
	w22 = random.uniform(-1,1)

	who1 = random.uniform(-1,1)
	who2 = random.uniform(-1,1)

	wih = [[w11,w12],
		[w21,w22]]
	


	who = [[who1,who2]]

	mat = np.array(inputs)
	mat2 = np.array(wih)
	print("-"*60)
	print(mat)
	print(mat2)

	sig = np.vectorize(f)
	resx = np.multiply(mat2,mat)
	print("-"*60)
	print("multiply weigths by inputs Layer - 1")
	print("-"*60)
	print(resx)

	resx = np.sum(resx,axis=1)
	print("-"*60)
	print("Sum of product Layer - 1")
	print("-"*60)
	print(resx)

	resx+=[random.uniform(-1,1),random.uniform(-1,1)]
	print("-"*60)
	print("Sum of bias Layer 1")
	print("-"*60)
	print(resx)

	resx = sig(resx)
	print("-"*60)
	print("Apply function sigmoid to Layer 1")
	print("-"*60)
	print(resx)

	out = np.multiply(who,resx)

	out = np.sum(resx,axis=0)
	out+random.uniform(-1,1)
	out = sig(out)
	#print(out)
	return out

def train(inputsT,targets):
	out = feedForward(inputs)
	#calcular error
	#error = target - out
	error = targets-out
	#print(error)


for i in inputs:
	train(inputs,targets)
	#feedForward(inputs)
#neurons weigths



