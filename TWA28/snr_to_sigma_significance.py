from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

def snr_to_sigma_significance(snr):
    """
    Convert a signal-to-noise ratio (SNR) to sigma significance via a log p-value.

    Parameters:
    snr (float): The signal-to-noise ratio.

    Returns:
    float: The sigma significance.
    """
    # Calculate the log of the p-value from the SNR
    log_p_value = norm.logsf(snr)
    
    # Convert the log p-value to sigma significance (z-score)
    sigma = norm.isf(np.exp(log_p_value))
    
    return sigma

# Example usage
snr_arr = np.linspace(1, 25, 10)
sigma_arr = [snr_to_sigma_significance(s) for s in snr_arr]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(snr_arr, sigma_arr, marker='o')

# plot a diagonal line for reference
x = np.linspace(0, 25, 100)
y = x
ax.plot(x, y, linestyle='--', color='gray')
ax.set_xlabel('Signal-to-Noise Ratio (SNR)')
ax.set_ylabel('Sigma Significance')
ax.set_title('SNR to Sigma Significance Conversion')
plt.show()
