from bayes_opt import BayesianOptimization
import numpy as np
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Instruction, Parameter

# Generate the BPSK signal sequence
def generate_bpsk_signal(num_symbols):
    symbols = np.random.randint(0, 2, size=num_symbols)  # Randomly generate a binary sequence, 0 or 1
    bpsk_signal = 2 * symbols - 1  # Convert the binary sequence to BPSK symbols: 0 becomes -1, 1 becomes 1
    return np.array(symbols).reshape(1, -1), bpsk_signal.reshape(-1, 1)  # Reshape as column vectors and return

# Generate AWGN noise power based on SNR
def generate_awgn_noise(signal_power, snr_dB, num_samples):
    # Calculate the noise power
    noise_power = signal_power / (10 ** (snr_dB / 10))

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=num_samples)

    # Reshape the noise into a column vector
    noise = noise.reshape(-1, 1)

    return noise

# Define channel matrix
h_4 = np.array([[0.92, 0.11, 0.09, 0.02],
              [0.11, 0.92, 0.03, 0.06],
              [0.09, 0.03, 0.98, 0.01],
              [0.02, 0.06, 0.01, 0.93]])
h_41 = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Define another channel matrix
h6 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03,0.01],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03],
              [0.03, 0.02, 0.02, 0.04, 0.98, 0.03],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.99]])

h6_1 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03,0.01],
              [0.11, 0.92, 0.03, 0.01, 0.02, 0.01],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03],
              [0.01, 0.01, 0.02, 0.01, 0.98, 0.03],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.99]])

h6_3 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03,0.01],
              [0.11, 0.92, 0.03, 0.32, 0.01, 0.01],
              [0.09, 0.03, 0.98, 0.01, 0.05, 0.02],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03],
              [0.21, 0.02, 0.02, 0.04, 0.98, 0.03],
              [0.01, 0.04, 0.23, 0.01, 0.05, 0.99]])

# Define yet another channel matrix
h8 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03, 0.01, 0.09, 0.02],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01, 0.03, 0.06],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02, 0.09, 0.03],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03, 0.02, 0.06],
              [0.03, 0.02, 0.02, 0.04, 0.98, 0.03, 0.03, 0.02],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.99, 0.01, 0.04],
              [0.02, 0.11, 0.09, 0.02, 0.03, 0.01, 0.99, 0.02],
              [0.11, 0.02, 0.03, 0.06, 0.02, 0.01, 0.03, 0.96],])

h8_1 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03, 0.01, 0.12, 0.02],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01, 0.03, 0.06],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02, 0.09, 0.03],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03, 0.02, 0.06],
              [0.03, 0.02, 0.22, 0.04, 0.98, 0.03, 0.03, 0.02],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.89, 0.01, 0.04],
              [0.02, 0.01, 0.09, 0.02, 0.03, 0.01, 0.99, 0.02],
              [0.01, 0.02, 0.03, 0.06, 0.02, 0.01, 0.03, 0.96],])
h8_2 = np.array([[0.92, 0.11, 0.09, 0.42, 0.03, 0.01, 0.09, 0.02],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01, 0.03, 0.06],
              [0.09, 0.03, 0.88, 0.01, 0.07, 0.02, 0.09, 0.03],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03, 0.02, 0.06],
              [0.03, 0.02, 0.02, 0.04, 0.98, 0.03, 0.03, 0.02],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.79, 0.01, 0.04],
              [0.32, 0.11, 0.19, 0.02, 0.03, 0.01, 0.99, 0.02],
              [0.11, 0.02, 0.13, 0.06, 0.02, 0.22, 0.03, 0.96],])

# Generate Gaussian white noise with signal power of 1 and SNR of 10 dB, with 4 samples
snr_dB = 10
num_samples = 4, 6, 8
signal_power = 1

#h = np.eye(num_samples)

noise = generate_awgn_noise(signal_power, snr_dB, num_samples)

transmitter_symbols, transmitter_signals  = generate_bpsk_signal(num_samples)

received_signals = np.dot(h, transmitter_signals) + noise

transmitter_symbols, transmitter_signals  = generate_bpsk_signal(num_samples)

received_signals = np.dot(h, transmitter_signals) + noise
