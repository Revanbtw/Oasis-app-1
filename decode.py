import soundfile as sf
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal
from typing import *
from numpy.typing import *

WORDS=['J', 'C']
with open("fichiers/mots-corr.txt", "r") as f:
    WORDS.extend(f.read().split("\n"))
    WORDS= list(filter(lambda x: len(x) > 0, WORDS))
WORDS = list(set(WORDS))    # On enlève les doublons (c'est pas très efficace désolé,,)

BAND = (501, 526)
SAMPLE_RATE = 8000
PADDED_SIZE = 16000
LEN_LETTRE = 2000           # Longueur d'une lettre
LEN_INTER = 500             # Espace entre les lettres
MARGE_FREQUENCE = 50        # L'espace autour de la bande de fréquence dans lequel chercher
AMP_PARASITE_MIN = 200      # Amplitude d'une fréquence parasite

def compute_letter_distance(c1: str, c2: str):
    if c1 == ' ' and c2 == ' ':
        return 0
    elif c1 == ' ' or c2 == ' ':
        return 26
    else:
        return abs(ord(c1) - ord(c2))

def compute_word_distance(word_1: str, word_2: str) -> int:
    """
    Calcule la distance entre le mot 1 et le mot 2, si tant est que ces deux mots
    soient en majuscules et sans accents. Ils peuvent être de longueur différente.

    Si l'un des mots est plus long que l'autre, le mot plus court est complété
    par des espaces.

    La distance d'une lettre à une autre est définie par leur distance dans
    l'alphabet. (Ex: d(A, E) = 4 et d(E, E) = 0)
    La distance d'une lettre avec un espace vaut toujours 26.

    La distance entre deux mots est calculée en effectuant la somme des distances
    de toutes les lettres.
    """
    if len(word_1) == 0 or len(word_2) == 0:
        return -1

    if len(word_1) > len(word_2):
        word_2 = word_2.ljust(len(word_1))
    else:
        word_1 = word_1.ljust(len(word_2))

    dist = 0
    for c1, c2 in zip(word_1, word_2):
        dist += compute_letter_distance(c1, c2)
    
    return dist

def find_closest_word(word: str, multiple = 0) -> str | List[str]:
    """
    Trouve le mot français qui minimise sa distance avec word.
    """
    candidates = sorted(WORDS, key= lambda w: compute_word_distance(w, word))
    if multiple > 0:
        return candidates[:multiple]
    return candidates[0]

def find_closest_words(string: str, multiple = 0) -> str | List[str] :
    """
    Utilise find_closest_word sur tous les mots d'une phrase.
    """
    words = string.split(' ')
    res = []
    for w in words:
        if w != "" and w != " ":
            w_res = find_closest_word(w, multiple)
            res.append(w_res)
    
    if multiple > 0:
        return [" ".join(ws) for ws in res]
    
    return " ".join(res)

def freq_to_letter(f):
    """
    Cette fonction trouve la lettre associée à la fréquence donnée en entrée.
    """
    i = max(min(round(f - BAND[0]), 25), 0)
    return chr(ord('A') + i)

def indice_to_freq(i: int, length: int, samplerate: int) -> float:
    """
    Cette fonction traduit un indice dans une TFD en la fréquence associée.
    """
    return i/length*samplerate

def freq_to_indice(f: float, length: int, samplerate: int) -> int:
    """
    Cette fonction est l'inverse de la fonction indice_to_freq et renvoie
    un entier.
    """
    return int(f*length//samplerate)

def trouve_max_deriv(donnees: np.ndarray, start: int, end: int) -> Optional[NDArray]:
    """
    Trouve les maximums locaux des données données en entrée en relevant les
    points de changement de signe de la dérivée.
    """
    deriv = np.convolve(donnees, np.array([1, -1]))
    candidats = []

    for i in range(start+ 1, end):
        if deriv[i-1]>0 and deriv[i]<0:
            candidats.append(i)
    
    if candidats :
        return np.array(sorted(candidats, key= lambda i: donnees[i])) - 1
    else:
        return None

################################### README #####################################
# Ce programme inclut une fonction find_closest_words qui permet de trouver,
# pour chaque mot du message, le mot de la langue française qui lui ressemble le
# plus. Vous pourrez l'essayer si vous le souhaitez, bien que cela soit hors du
# sujet de l'exercice.
# On peut également, avec l'argument multiple= n, montrer le top n des mots les
# plus proches. C'est un assistant de décryptage, si on veut.
################################################################################

def decode(sig: NDArray):
    N = len(sig)
    samplerate = SAMPLE_RATE
    jambon = np.blackman(LEN_LETTRE)

    # On détermine le maximum des amplitudes du bruit
    tfd = fft.fft(sig[:LEN_LETTRE]*jambon, PADDED_SIZE)
    plt.show()
    maxi_bruit = np.max(
        np.abs(tfd[freq_to_indice(1000, PADDED_SIZE, samplerate): freq_to_indice(2000, PADDED_SIZE, samplerate)])
    )

    print(maxi_bruit)

    i = 0
    res = ""

    while i + LEN_LETTRE < N :
        tfd = fft.fft(sig[i : i + LEN_LETTRE]*jambon, PADDED_SIZE)    # On fait la tfd de la lettre

        i_inf = freq_to_indice(BAND[0], PADDED_SIZE, samplerate)
        i_sup = freq_to_indice(BAND[1], PADDED_SIZE, samplerate)

        m = trouve_max_deriv(
            np.abs(tfd[:PADDED_SIZE//2]),
            i_inf - MARGE_FREQUENCE,
            i_sup + MARGE_FREQUENCE
        )

        if type(m) != None :
            # On enlève tous les indices indésirables, i.e. qui sont du bruit ou des parasites
            m = m[(AMP_PARASITE_MIN >= np.abs(tfd[m])) & ((maxi_bruit + 5) <= np.abs(tfd[m]))] # type: ignore
            # m = m[(i_inf < m) & (m < i_sup)]  # Cette ligne est facultative
            if len(m) > 0:
                res += freq_to_letter(indice_to_freq(m[0], PADDED_SIZE, samplerate))
            else:
                res += " "
        else :
            res += " "

        i += LEN_INTER + LEN_LETTRE

    return res

if __name__ == "__main__":
    sig, samplerate = sf.read("fichiers/mess_difficile.wav")
    decoded = decode(sig)
    print(f"{decoded}")
    print(f"{find_closest_words(decoded, multiple=5)}\n")