from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
import numpy as np
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pylab
import matplotlib.colors as mcolors
import copy
import networkx as nx
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit_aer import AerSimulator, noise
from qiskit.utils import QuantumInstance
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from scipy.optimize import minimize
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import CircuitStateFn, PauliExpectation
from qiskit.algorithms.gradients import FiniteDiffEstimatorGradient
from main import h, signal_power, snr_dB, num_samples, generate_bpsk_signal,generate_awgn_noise


def store_intermediate_result1(eval_count, parameters, mean, std):
    counts1.append(eval_count)
    values1.append(mean)

def qaoa_circuit_for_COBYLA(num_qubits, depth_p, params, matrix_A, vector_b):

    gamma_params = params[:depth_p]
    beta_params =  params[depth_p:]

    pure_qaoa_for_COBYLA = QuantumCircuit(num_qubits)

    # Apply Hadamard gate to each qubit
    for qubit in range(num_qubits):
        pure_qaoa_for_COBYLA.h(qubit)
    pure_qaoa_for_COBYLA.barrier()
    for p in range(depth_p):
        # Iterate through all qubit combinations and add an RZZ gate for each pair of qubits
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pure_qaoa_for_COBYLA.rzz(matrix_A[i][j] * gamma_params[p], i, j)
        pure_qaoa_for_COBYLA.barrier()
        for i in range(num_qubits):
            pure_qaoa_for_COBYLA.rz(-1 * vector_b[i] * gamma_params[p], i)

        for i in range(num_qubits):
            pure_qaoa_for_COBYLA.rx(beta_params[p], i)
        pure_qaoa_for_COBYLA.barrier()
    return pure_qaoa_for_COBYLA

def pure_qaoa_circuit_for_Bayesian(num_qubits, depth_p, params, matrix_A, vector_b):

    gamma_params = params[:depth_p]
    beta_params =  params[depth_p:]

    pure_qaoa_for_Bayesian = QuantumCircuit(num_qubits)

    # Apply a Hadamard gate to each qubit
    for qubit in range(num_qubits):
        pure_qaoa_for_Bayesian.h(qubit)
    pure_qaoa_for_Bayesian.barrier()
    for p in range(depth_p):
        # Iterate through all qubit combinations and add an RZZ gate for each pair of qubits
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pure_qaoa_for_Bayesian.rzz(matrix_A[i][j] * gamma_params[p], i, j)
        pure_qaoa_for_Bayesian.barrier()
        for i in range(num_qubits):
            pure_qaoa_for_Bayesian.rz(-1 * vector_b[i] * gamma_params[p], i)

        for i in range(num_qubits):
            pure_qaoa_for_Bayesian.rx(beta_params[p], i)
        pure_qaoa_for_Bayesian.barrier()
    return pure_qaoa_for_Bayesian


qubits = num_samples

depth_p = 4, 6, 8

# Define A = h^H * h
A = np.dot(np.conjugate(h).T, h)  / 100
# Set the diagonal elements to zero
np.fill_diagonal(A, 0)

# Generate BPSK signal and add noise for transmission
noise = generate_awgn_noise(signal_power, snr_dB, num_samples)
transmitter_symbols, transmitter_signals = generate_bpsk_signal(num_samples)
received_signals = np.dot(h, transmitter_signals) + noise

# Received signal b = y.T * H
b = np.dot(received_signals.T, h) # Local magnetic fields
b = b.flatten()

# Generate QAOA circuit with parameters based on information
pure_qaoa_for_COBYLA = pure_qaoa_circuit_for_COBYLA(qubits, depth_p,[np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi,np.pi, np.pi,np.pi,np.pi,np.pi  ], A, b)
pure_qaoa_for_COBYLA.draw(output='mpl')
plt.show()

# Define the QAOA circuit to solve the problem's Hamiltonian Hf
# Initialize the Pauli string list
pauli_strings = []
paulis = []
coeffs = []
# Add local magnetic field terms
for i in range(qubits):
    pauli_strings.append((("I" * i) + "Z" + ("I" * (qubits - i - 1)), -1 * b[( qubits - 1 )-i]))
    paulis.append(("I" * i) + "Z" + ("I" * (qubits - i - 1)))
    coeffs.append(-1 * b[( qubits - 1 )- i])
# Add interaction terms
for i in range(qubits):
    for j in range(i + 1, qubits):
        pauli_strings.append((("I" * i) + "Z" + ("I" * (j - i - 1)) + "Z" + ("I" * (qubits - j - 1)), A[( qubits - 1 )- i][( qubits - 1 ) - j]))
        paulis.append(("I" * i) + "Z" + ("I" * (j - i - 1)) + "Z" + ("I" * (qubits - j - 1)))
        coeffs.append(A[( qubits - 1 ) - i][( qubits - 1 ) - j])
#print(pauli_strings)
#print(paulis)
#print(coeffs)
hamiltonian = SparsePauliOp.from_list(pauli_strings)

def objective_function_for_COBYLA(params):

    params = params
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    # Simulation of the quantum circuit
    sim_noise = AerSimulator(noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_CNNs_noise = transpile(pure_qaoa_for_Bayesian, sim_noise)

    result = sim_noise.run(circ_CNNs_noise, shots=2048).result()
    '''
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # Compile the circuit to fit the selected backend
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # Run the circuit and get the result
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    '''
    # Get the measurement results
    counts_circuit = result.get_counts()

    # Initialize the expectation value
    expectation = 0

    # Iterate over each term in the Hamiltonian
    for pauli, coeff in zip(paulis, coeffs):
        # Initialize the expectation value for the current Hamiltonian term
        term_expectation = 0

        # Iterate over each ground state in the samples
        for basis_state, count in counts_circuit.items():
            # Compute the expectation value for the current ground state and weight it
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # If the Hamiltonian term is the identity matrix, the expectation value is the frequency of the ground state
            else:
                basis_state_contrib = 1  # Contribution of the ground state initialized to 1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1  # If the Hamiltonian term is not the identity matrix, compute the contribution based on the corresponding bit value of the ground state
                term_expectation += basis_state_contrib * count  # Multiply the contribution of the current ground state by its frequency and accumulate it to the expectation value of the current Hamiltonian term

        # Multiply the expectation value of the current Hamiltonian term by its coefficient and accumulate it to the total expectation value
        expectation += coeff * term_expectation

    # Normalize the expectation value
    total_counts = sum(counts_circuit.values())
    expectation /= total_counts
    values.append(expectation)

    return expectation

'''
optimizer = COBYLA(maxiter=100)

estimator = Estimator()   # Since VQE uses wavefunction estimation, the evaluator can be set to None."
'''

initial_params = np.random.uniform(0, 2*np.pi, 2 * depth_p)

def objective_function_for_Bayesian(params_0, params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15):

    params = [params_0, params_1,params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15]
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # Compile the circuit to fit the selected backend
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # Run the circuit and get the result
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()

    # Get the measurement results
    counts_circuit = result.get_counts()

    # Initialize the expectation value
    expectation = 0

    # Iterate over each term in the Hamiltonian
    for pauli, coeff in zip(paulis, coeffs):
        # Initialize the expectation value for the current Hamiltonian term
        term_expectation = 0

        # Iterate over each ground state in the samples
        for basis_state, count in counts_circuit.items():
            # Compute the expectation value for the current ground state and weight it
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # If the Hamiltonian term is the identity matrix, the expectation value is the frequency of the ground state
            else:
                basis_state_contrib = 1  # Contribution of the ground state initialized to 1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1 # If the Hamiltonian term is not the identity matrix, compute the contribution based on the corresponding bit value of the ground state
                term_expectation += basis_state_contrib * count  # Multiply the contribution of the current ground state by its frequency and accumulate it to the expectation value of the current Hamiltonian term

        # Multiply the expectation value of the current Hamiltonian term by its coefficient and accumulate it to the total expectation value
        expectation += coeff * term_expectation


def objective_function_for_COBYLA1(params):

    params = params
    pure_qaoa_for_Bayesian = pure_qaoa_circuit_for_Bayesian(qubits, depth_p, params, A, b)
    pure_qaoa_for_Bayesian.measure_all()

    # Simulation of the quantum circuit
    sim_noise = AerSimulator(noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_CNNs_noise = transpile(pure_qaoa_for_Bayesian, sim_noise)

    result = sim_noise.run(circ_CNNs_noise, shots=2048).result()
    '''
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = 2048

    # Compile the circuit to fit the selected backend
    compiled_circuit = transpile(pure_qaoa_for_Bayesian, backend)

    # Run the circuit and get the result
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    '''
    # Get the measurement results
    counts_circuit = result.get_counts()

    # Initialize the expectation value
    expectation = 0

    # Iterate over each term in the Hamiltonian
    for pauli, coeff in zip(paulis, coeffs):
        # Initialize the expectation value for the current Hamiltonian term
        term_expectation = 0

        # Iterate over each ground state in the samples
        for basis_state, count in counts_circuit.items():
            # Compute the expectation value for the current ground state and weight it
            if all(pauli_char == 'I' for pauli_char in pauli):
                term_expectation += count  # If the Hamiltonian term is the identity matrix, the expectation value is the frequency of the ground state
            else:
                basis_state_contrib = 1  # Contribution of the ground state initialized to 1
                for i, pauli_char in enumerate(pauli):
                    if pauli_char == 'Z' and basis_state[i] == '1':
                        basis_state_contrib *= -1  # If the Hamiltonian term is not the identity matrix, compute the contribution based on the corresponding bit value of the ground state
                term_expectation += basis_state_contrib * count  # Multiply the contribution of the current ground state by its frequency and accumulate it to the expectation value of the current Hamiltonian term

        # Multiply the expectation value of the current Hamiltonian term by its coefficient and accumulate it to the total expectation value
        expectation += coeff * term_expectation

    # Normalize the expectation value
    total_counts = sum(counts_circuit.values())
    expectation /= total_counts
    values1.append(expectation)
