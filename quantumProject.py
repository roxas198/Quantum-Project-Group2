from qiskit import QuantumCircuit, assemble, Aer

#Number of qubits is 4
nQubit = 4

#Initialize the circuit for Grover's Algorithm
groverAlgorithm = QuantumCircuit(nQubit)

#Apply Hadamard transform to all qubits. This gives the quantum superposition for ampltiude amplification
#That is, gives the state of 1/sqrt(N) sum_{x=0}^{N-1} |x>, where N = 2^nQubit
groverAlgorithm.h(range(nQubit))

#Prints out circuit
print(groverAlgorithm)