import numpy as np
from main import h, signal_power, snr_dB, num_samples, generate_bpsk_signal,generate_awgn_noise

noise = generate_awgn_noise(signal_power, snr_dB, num_samples)

transmitter_symbols, transmitter_signals = generate_bpsk_signal(num_samples)

received_signals = np.dot(h, transmitter_signals) + noise

print(transmitter_symbols)
#print(np.dot(h, transmitter_signal))
#print(noise)
print(received_signals)

# ML Detection (assuming knowledge of H)
estimated_transmit_signal = np.dot(np.linalg.inv(h), received_signals)
#print(estimated_transmit_signal)

# Demodulation
threshold = 0  # Threshold
demodulated_symbols = np.where(estimated_transmit_signal > threshold, 1, -1)

# Decoding
decoded_bits = ((demodulated_symbols + 1) // 2).T  # Map 1 to 1, -1 to 0

#print(decoded_bits)

# Compare two row vectors, return a boolean array indicating whether corresponding elements are the same
element_wise_comparison = transmitter_symbols == decoded_bits
