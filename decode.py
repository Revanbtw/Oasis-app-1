import soundfile as sf
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

BAND = (501, 526)
PADDED_SIZE = 16000
ECART_MOY_MIN = 10

def freq_to_letter(f):
    i = max(min(round(f - BAND[0]), 25), 0)
    return chr(ord('A') + i)

def max_freq(tf, samplerate):
    m = np.argmax(np.abs(tf[:len(tf)//2]))
    return m/len(tf)*samplerate, m

# def filt_amps(tf, samplerate):


def decode(fp):
    sig, samplerate = sf.read(fp)
    N = len(sig)
    N_signaux = 20 # Nombre de sons à mesurer
    len_signal = N//N_signaux

    # On calcule les différentes fréquences en prenant un échantillon de 16000 échantillons au millieu de chaque signal
    for i in range(N_signaux):
        tfd = fft.fft(sig[i*len_signal : (i+1)*len_signal], PADDED_SIZE)
        m, i = max_freq(tfd, samplerate)
        moy = np.average(tfd)
        if np.abs(np.abs(tfd[i]) - np.abs(moy)) >= ECART_MOY_MIN and BAND[0] - 1 <= m <= BAND[1] + 1:
            print(freq_to_letter(m), end="")
        else:
            print("_", end="")

decode("fichiers/mess.wav")
print("")