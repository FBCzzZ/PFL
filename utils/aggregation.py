import numpy as np


def aggregation_spec(w_FFT_list):
    w_FFT_np = np.vstack(w_FFT_list)
    w_FFT_glob = np.mean(w_FFT_np, axis=0)

    median = np.median(w_FFT_glob)
    w_FFT_glob[w_FFT_glob > median] = 0

    return w_FFT_glob