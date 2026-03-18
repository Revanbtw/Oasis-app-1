import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy import signal


a, b = signal.butter(5, [2000, 5000], btype="bandpass", fs=16000)

w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.show()
