from Neuron import Neuron
import random
import numpy as np
from math import exp
from Network import Network


#entradas que coincidan con el parametro de feedforward
n = Network(2,2,1)

inputs = [[1,1],[0,0],[1,0],[0,1]]
ans = [0,0,1,1]

for x in xrange(1,10000):
	nu = random.randint(0,3)
	a = inputs[nu]
	b = ans[nu]
	n.feedForward(a,b)



print("GUESS")
print(n.feedForward([[1,0]],[1]))
print(n.feedForward([[1,1]],[0]))
print(n.feedForward([[0,1]],[1]))
print(n.feedForward([[0,0]],[0]))




