import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

frequencies = np.array([1.5, 1.4, 1.35, 1.34, 1.335, 1.33, 1.327, 1.326, 1.325, 1.323, 1.32, 1.318044, 1.31, 1.3, 1.2, 1.1])
displacements = np.array([0.002, 0.005, 0.016, 0.027, 0.035, 0.047, 0.052, 0.052, 0.053, 0.051, 0.042, 0.037, 0.018, 0.012, 0.003, 0.002])


# Smooth the displacement data using a Gaussian filter
sigma = 1.0  # Adjust sigma for more or less smoothing
smoothed_displacements = gaussian_filter1d(displacements, sigma=sigma)

# Find the index of the peak (resonant frequency) in the smoothed data
peak_indices, _ = find_peaks(smoothed_displacements)
resonant_frequency = frequencies[peak_indices[np.argmax(smoothed_displacements[peak_indices])]]

# Plot the original and smoothed frequency curves
plt.plot(frequencies, displacements, label='Original Displacement vs Frequency', linestyle='dotted', color='gray')
plt.plot(frequencies, smoothed_displacements, label='Smoothed Displacement vs Frequency')

# Highlight the resonant frequency on the plot
plt.scatter(resonant_frequency, np.max(smoothed_displacements), color='red', zorder=5, label=f'Resonant Frequency: {resonant_frequency} Hz')

# Labels and title
plt.title('Smoothed Frequency Response Curve')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Displacement (m)')
plt.legend()

# Show the plot
plt.show()

print(f"Resonant Frequency: {resonant_frequency} Hz")
