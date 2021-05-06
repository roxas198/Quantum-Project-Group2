from qiskit import QuantumCircuit, assemble, Aer, extensions
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import math

#Number of qubits is 4
nQubit = 4

#Gives a permutation of a list from 0 to N-1.
L = np.random.permutation(2**nQubit)
print(L)
#Randomly select index from 0 to N-1
y = np.random.randint(2**nQubit)
print(y)
print(math.ceil(22.5*math.sqrt(2**nQubit) + 1.4 * (math.log(2**nQubit, 2))**2))
#for i in range(math.ceil(22.5*math.sqrt(2**nQubit) + 1.4 * (math.log(2**nQubit, 2))**2)):
matrix = np.zeros((2**nQubit,2**nQubit))

for i in range(2**nQubit):
    if (L[i] >= L[y]):
        matrix[i][i] = 1
    else:
        matrix[i][i] = -1

oracle = extensions.UnitaryGate(matrix)
#Initialize the circuit for Grover's Algorithm
groverAlgorithm = QuantumCircuit(nQubit, nQubit)

#Apply Hadamard transform to all qubits. This gives the quantum superposition for ampltiude amplification
#That is, gives the state of 1/sqrt(N) sum_{x=0}^{N-1} |x>, where N = 2^nQubit
groverAlgorithm.h(range(nQubit))

#Applies the oracle
groverAlgorithm.unitary(oracle, range(nQubit))

#Applies Grover's Algorithm
groverAlgorithm.h(range(nQubit))
groverAlgorithm.x(range(nQubit))
groverAlgorithm.i(range(nQubit-1))
groverAlgorithm.h(nQubit-1)
#Need to make this a toffoli gate that uses nQubit-1 control bits on the final qubit
#groverAlgorithm.toffoli(0, 1, 2)
groverAlgorithm.mct([0,1,2], 3)
groverAlgorithm.i(range(nQubit-1))
groverAlgorithm.h(nQubit-1)
groverAlgorithm.x(range(nQubit))
groverAlgorithm.h(range(nQubit))
for i in range(nQubit):
    groverAlgorithm.measure(i,i)
#Prints out circuit
print(groverAlgorithm)

sim = Aer.get_backend('qasm_simulator')  # this is the simulator we'll use
qobj = assemble(groverAlgorithm)  # this turns the circuit into an object our backend can run
result = sim.run(qobj).result()  # we run the experiment and get the result from that experiment
# from the results, we get a dictionary containing the number of times (counts)
# each result appeared
counts = result.get_counts()
#print(counts)
indexResult = int(max(counts, key=counts.get), 2)
if (L[indexResult] < L[y]):
    y = indexResult
# and display it on a histogram
print(indexResult)
plot_histogram(counts)
plt.show()