import numpy as np
import matplotlib.pyplot as plt

K_vert_5x3 = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
 [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
 [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
 [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])

# Kernel size
freq_bins = 40   # vertical axis (frequency)
time_bins = 20   # horizontal axis (time)

# Create smooth frequency profile: almost flat, small gentle slope
freq_profile = np.linspace(1, 0.9, freq_bins)[:, np.newaxis]  # very gentle high→low

# Smooth time modulation: soft sine wave
time_profile = np.sin(np.linspace(0, np.pi, time_bins))[np.newaxis, :]

# Combine profiles to get 2D kernel
kernel = freq_profile * time_profile  # element-wise multiplication

# Normalize: zero-mean and unit-norm
kernel -= kernel.mean()
kernel /= np.linalg.norm(kernel) + 1e-12

# Visualize
plt.figure(figsize=(6,4))
plt.imshow(K_vert_5x3, aspect='auto', origin='lower', cmap='RdBu_r')
plt.colorbar(label='Amplitude')
plt.title("Smooth Flat-Frequency Kernel (40x20)")
plt.xlabel("Time bins")
plt.ylabel("Frequency bins")
plt.show()