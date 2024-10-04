import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# First set of data
frequencies1 = np.array([1.5, 1.4, 1.35, 1.34, 1.335, 1.33, 1.327, 1.326, 1.325, 1.323, 1.32, 1.318044, 1.31, 1.3, 1.2, 1.1])
displacements1 = np.array([0.002, 0.005, 0.016, 0.027, 0.035, 0.047, 0.052, 0.052, 0.053, 0.051, 0.042, 0.037, 0.018, 0.012, 0.003, 0.002])

# Second set of data
frequencies2 = np.array([1.4, 1.36, 1.34, 1.33, 1.325, 1.32, 1.318044, 1.31, 1.3])
displacements2 = np.array([0.004, 0.009, 0.015, 0.018, 0.018, 0.017, 0.016, 0.013, 0.01])

# Smoothing with Gaussian filter
sigma = 1.0
smoothed_displacements1 = gaussian_filter1d(displacements1, sigma=sigma)
smoothed_displacements2 = gaussian_filter1d(displacements2, sigma=sigma)

# Function to compute Q-factor
def compute_q_factor(frequencies, displacements, peak_displacement, resonant_frequency):
    half_power_level = peak_displacement / 2
    crossing_indices = np.where(np.diff(np.sign(displacements - half_power_level)))[0]
    print(crossing_indices)

    if len(crossing_indices) == 0:
        return np.nan, np.nan

    # Check for crossing on both sides
    lower_crossing = crossing_indices[crossing_indices < np.argmax(displacements)]
    higher_crossing = crossing_indices[crossing_indices > np.argmax(displacements)]

    if len(lower_crossing) > 0 and len(higher_crossing) > 0:
        # Both sides present, use the difference
        print("BOTH")
        delta_f = abs(frequencies[higher_crossing[0]] - frequencies[lower_crossing[-1]])
    elif len(lower_crossing) > 0:
        print("LOWER", resonant_frequency, frequencies[lower_crossing[-1]], half_power_level)
        # Only the lower side is present, double the distance
        delta_f = 2 * abs(resonant_frequency - frequencies[lower_crossing[-1]])
    elif len(higher_crossing) > 0:
        print("UPPER")
        # Only the higher side is present, double the distance
        delta_f = 2 * abs(frequencies[higher_crossing[0]] - resonant_frequency)
    else:
        # Neither side has crossings, return NaN
        delta_f = np.nan

    if delta_f is not np.nan:
        Q = (np.sqrt(3) * resonant_frequency) / delta_f
    else:
        Q = np.nan

    return Q, delta_f

# Finding peaks and computing Q-factors
peak_indices1, _ = find_peaks(smoothed_displacements1)
resonant_frequency1 = frequencies1[peak_indices1[np.argmax(smoothed_displacements1[peak_indices1])]]
peak_displacement1 = np.max(smoothed_displacements1)
Q1, delta_f1 = compute_q_factor(frequencies1, smoothed_displacements1, peak_displacement1, resonant_frequency1)

peak_indices2, _ = find_peaks(smoothed_displacements2)
resonant_frequency2 = frequencies2[peak_indices2[np.argmax(smoothed_displacements2[peak_indices2])]]
peak_displacement2 = np.max(smoothed_displacements2)
Q2, delta_f2 = compute_q_factor(frequencies2, smoothed_displacements2, peak_displacement2, resonant_frequency2)

# Plotting the data
plt.plot(frequencies1, displacements1, label='Original Set 1', linestyle='dotted', color='gray')
plt.plot(frequencies1, smoothed_displacements1, label='Smoothed Set 1', color='blue')
plt.scatter(resonant_frequency1, peak_displacement1, color='red', zorder=5,
            label=f'Set 1: f_res = {resonant_frequency1:.3f} Hz, Q = {Q1:.2f}')

plt.plot(frequencies2, displacements2, label='Original Set 2', linestyle='dotted', color='orange')
plt.plot(frequencies2, smoothed_displacements2, label='Smoothed Set 2', color='green')
plt.scatter(resonant_frequency2, peak_displacement2, color='purple', zorder=5,
            label=f'Set 2: f_res = {resonant_frequency2:.3f} Hz, Q = {Q2:.2f}')

# Labels and title
plt.title('Frequency Response Curves with Resonant Frequencies and Q-Factors')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Displacement (m)')
plt.legend()

# Output resonant frequencies and Q-factors
print(f"Resonant Frequency 1: {resonant_frequency1} Hz, Q-factor 1: {Q1}, Delta: {delta_f1}")
print(f"Resonant Frequency 2: {resonant_frequency2} Hz, Q-factor 2: {Q2}, Delta: {delta_f2}")

# Show the plot
plt.show()
