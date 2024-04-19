from bayes_opt import BayesianOptimization
import numpy as np
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Instruction, Parameter

# Generate the BPSK signal sequence
def generate_bpsk_signal(num_symbols):
    symbols = np.random.randint(0, 2, size=num_symbols)  # 随机生成二进制序列，0 或 1
    bpsk_signal = 2 * symbols - 1  # 将二进制序列转换为 BPSK 符号：0 变为 -1，1 变为 1
    return np.array(symbols).reshape(1, -1), bpsk_signal.reshape(-1, 1)  # 转换为列向量返回

# 根据SNR生成噪声功率
def generate_awgn_noise(signal_power, snr_dB, num_samples):
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_dB / 10))

    # 生成符合高斯分布的噪声
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=num_samples)

    # 将噪声变形为列向量
    noise = noise.reshape(-1, 1)

    return noise


# Define channel matrix
h = np.array([[0.92, 0.11, 0.09, 0.02],
              [0.11, 0.92, 0.03, 0.06],
              [0.09, 0.03, 0.98, 0.01],
              [0.02, 0.06, 0.01, 0.93]])

# Define channel matrix
h2 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03,0.01],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03],
              [0.03, 0.02, 0.02, 0.04, 0.98, 0.03],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.99]])

# Define channel matrix
h3 = np.array([[0.92, 0.11, 0.09, 0.02, 0.03, 0.01, 0.09, 0.02],
              [0.11, 0.92, 0.03, 0.06, 0.02, 0.01, 0.03, 0.06],
              [0.09, 0.03, 0.98, 0.01, 0.07, 0.02, 0.09, 0.03],
              [0.02, 0.06, 0.01, 0.93, 0.02, 0.03, 0.02, 0.06],
              [0.03, 0.02, 0.02, 0.04, 0.98, 0.03, 0.03, 0.02],
              [0.01, 0.04, 0.03, 0.01, 0.05, 0.99, 0.01, 0.04],
              [0.02, 0.11, 0.09, 0.02, 0.03, 0.01, 0.99, 0.02],
              [0.11, 0.02, 0.03, 0.06, 0.02, 0.01, 0.03, 0.96],])
# 生成信号功率为 1，信噪比为 10 dB 的高斯白噪声，共 4 个样本
snr_dB = 10
num_samples = 4
signal_power = 1

#h = np.eye(num_samples)

noise = generate_awgn_noise(signal_power, snr_dB, num_samples)

transmitter_symbols, transmitter_signals  = generate_bpsk_signal(num_samples)

received_signals = np.dot(h, transmitter_signals) + noise