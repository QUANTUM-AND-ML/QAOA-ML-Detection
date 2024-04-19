from bayes_opt import BayesianOptimization
import numpy as np
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Instruction, Parameter
from main import h, signal_power, snr_dB, num_samples, generate_bpsk_signal,generate_awgn_noise

'''
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
'''

wrong_number_of_bits = 0
number_of_signals_transmitted = 200000

# 设置每隔多少次循环打印一次计数器的值
print_interval = 100000

# 计数器初始化为 0
counter = 0

for i in range(number_of_signals_transmitted):
    noise = generate_awgn_noise(signal_power, snr_dB, num_samples)

    transmitter_symbols, transmitter_signals  = generate_bpsk_signal(num_samples)

    received_signals = np.dot(h, transmitter_signals) + noise

    print(transmitter_symbols)
    #print(np.dot(h, transmitter_signal))
    #print(noise)
    print(received_signals)

    # ML 检测（假设知道 H）
    estimated_transmit_signal = np.dot(np.linalg.inv(h), received_signals)
    #print(estimated_transmit_signal)

    # 解调
    threshold = 0  # 阈值
    demodulated_symbols = np.where(estimated_transmit_signal > threshold, 1, -1)

    # 译码
    decoded_bits = ((demodulated_symbols + 1) // 2).T  # 将 1 映射为 1，-1 映射为 0

    #print(decoded_bits)

    # 比较两个行向量，返回一个布尔数组，表示对应位置上元素是否相同
    element_wise_comparison = transmitter_symbols == decoded_bits

    # 统计布尔数组中为 False（不相同）的元素数量
    num_different_elements = np.sum(~element_wise_comparison)

    wrong_number_of_bits = num_different_elements + wrong_number_of_bits

    counter += 1
    # 如果计数器能被 print_interval 整除，则打印计数器的值
    if counter % print_interval == 0:
        print("当前循环次数：", counter)

print('误码率BER：', wrong_number_of_bits / ( num_samples * number_of_signals_transmitted))