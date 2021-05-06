#!/usr/bin/env python
# coding: utf-8

# In[341]:


# Header files importing all the packages

import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

from qiskit.visualization import plot_histogram


# In[342]:


H_gate = np.array([[1,1], [1,-1]])/math.sqrt(2)  # Hadamard gate
initState = np.array([[1],[0]]) # |0> vector 
N=16
L=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]


# In[343]:


# This function converts every integer to its 4-bit binary representation

def convertToBits (num):
    x = bin(num)
    x = x[2:]
    num1 = [0]*(int(math.log2(N)))
    for i in range(len(x)):
        num1[int(math.log2(N))-i-1] = int(x[len(x)-i-1])
    return num1


# In[344]:


def uniformly_at_random():
    
    # This will be taken care of later
    
    return 4


# In[355]:


# The reversible circuit for comparing the magnitude of 2 4-bit numbers

def magnitude_comparator(l_n,l_y):
    and_operation_anot_b = [0]*(int(math.log2(N)))
    and_operation_bnot_a = [0]*(int(math.log2(N)))
    for i in range(int(math.log2(N))):
        and_operation_anot_b[i] = invertBits(l_n[int(math.log2(N))-i-1]) & l_y[int(math.log2(N))-i-1]
        and_operation_bnot_a[i] = invertBits(l_y[int(math.log2(N))-i-1]) & l_n[int(math.log2(N))-i-1]
    
    temp = [0]*(int(math.log2(N))-1)
    for i in range(3):
        temp[i] = invertBits(and_operation_anot_b[int(math.log2(N))-i-1] | and_operation_bnot_a[int(math.log2(N))-i-1])
    
    param2 = temp[0] & (and_operation_anot_b[int(math.log2(N))-2])
    param3 = temp[0] & temp[1] & (and_operation_anot_b[int(math.log2(N))-3])
    param4 = temp[0] & temp[1] & temp[2] &  (and_operation_anot_b[0])
    
    return (and_operation_anot_b[int(math.log2(N))-1] | param2 | param3 | param4 )
        


# In[356]:


# Function to generate the Oracle to be used. (Implemeted based on the comparison of the magnitude of the numbers)
# U|n> = |n> if L[n] >= L[y] and U|n> = -|n> if L[n] < L[y].

def Oracle(y):
    U = np.zeros((N,N))
    l_y = convertToBits(L[y])
    for idx,n in enumerate(L):
        check = magnitude_comparator(convertToBits(n),l_y)
        if (check==1):
            U[idx,idx]=-1
        else:
            U[idx,idx]=1
#             print ("Printing else:",n)
    return U


# In[357]:


def generatingList():
    return ""
#     In this segment the value of L will be generated uniformly at random


# In[358]:


def tensorProduct(matrices):
    prod = matrices[0]
    for i in matrices[1:]:
        prod = np.kron(prod,i)  ## np.kron stands for Kronecker product, the official name for tensor product
    return prod


# In[359]:


# Function to perform the NOT-operation

def invertBits(num):  
    if num==0:
        return 1
    else:
        return 0


# In[360]:


y = uniformly_at_random();
# psi_0 = np.dot(tensorProduct(listGates_H))
U = Oracle(y)
print ("The oracle for the algorithm is: ")
print (U)


# In[ ]:




