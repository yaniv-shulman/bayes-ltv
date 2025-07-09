from typing import List, Tuple, Dict

import numpy as np
from scipy.signal import welch
from tqdm import tqdm


def calculate_psd_stats_db(
    good_pairs: List[Tuple[np.ndarray, np.ndarray]],
    fs: float,
    nperseg: int = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calculate the mean and standard deviation of the PSD in dB for each virtual source
    and receiver using only the good pairs. The mean of power in dB is computed for each segment,
    then aggregated in the dB domain.

    Args:
        good_pairs: A list of tuples (source_segment, receiver_segment).
        fs: The sampling frequency (Hz).
        nperseg: Length of each segment for Welch's method. Defaults to the length of the segment.

    Returns:
        A tuple of two dictionaries:
            - source_stats_db: Dictionary with keys 'freq', 'psd_mean_db', and 'psd_std_db' for the source.
            - receiver_stats_db: Dictionary with keys 'freq', 'psd_mean_db', and 'psd_std_db' for the receiver.
    """
    if not good_pairs:
        raise ValueError("The good_pairs list is empty.")

    # Default nperseg to the segment length (assuming all segments are the same length)
    if nperseg is None:
        nperseg = good_pairs[0][0].shape[0]
    else:
        seg_size = good_pairs[0][0].shape[0]

        if nperseg > seg_size:
            raise ValueError("nperseg > segment length")

    psd_source_list_db = []
    psd_receiver_list_db = []
    freq = None
    epsilon = 1e-12  # To prevent log of zero

    # Loop over each pair and compute the PSD in dB using Welch's method
    for seg_source, seg_receiver in good_pairs:
        freq, psd_source = welch(seg_source, fs=fs, nperseg=nperseg)
        _, psd_receiver = welch(seg_receiver, fs=fs, nperseg=nperseg)

        # Convert PSD to dB for each segment
        psd_source_db = 10 * np.log10(np.maximum(psd_source, epsilon))
        psd_receiver_db = 10 * np.log10(np.maximum(psd_receiver, epsilon))

        psd_source_list_db.append(psd_source_db)
        psd_receiver_list_db.append(psd_receiver_db)

    # Stack the PSD estimates so that each row corresponds to one segment (in dB)
    psd_source_arr_db = np.vstack(psd_source_list_db)
    psd_receiver_arr_db = np.vstack(psd_receiver_list_db)

    # Compute mean and std directly in the dB domain for each frequency bin
    source_psd_mean_db = np.mean(psd_source_arr_db, axis=0)
    source_psd_std_db = np.std(psd_source_arr_db, axis=0)
    receiver_psd_mean_db = np.mean(psd_receiver_arr_db, axis=0)
    receiver_psd_std_db = np.std(psd_receiver_arr_db, axis=0)

    source_stats_db = {
        "freq": freq,
        "psd_mean_db": source_psd_mean_db,
        "psd_std_db": source_psd_std_db,
    }

    receiver_stats_db = {
        "freq": freq,
        "psd_mean_db": receiver_psd_mean_db,
        "psd_std_db": receiver_psd_std_db,
    }

    return source_stats_db, receiver_stats_db


def _whiten(sig: np.ndarray, eps: float) -> np.ndarray:
    """
    Spectrally whiten a signal by normalizing its FFT coefficients.

    Args:
        sig: The input signal to be whitened, a numpy array.
        eps: A small constant to avoid division by zero in the normalization.

    Returns:

    """
    spec: np.ndarray = np.fft.rfft(sig)
    mag: np.ndarray = np.abs(spec)
    spec /= mag + eps
    return np.fft.irfft(spec, n=sig.size)


def spectral_whiten_pairs(
    pairs: List[Tuple[np.ndarray, np.ndarray]], eps: float = 1e-16
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Applies spectral whitening to each signal in each pair.

    Spectral whitening is achieved by computing the FFT of the signal, adding a small constant (eps) to avoid division
    by zero, dividing by the magnitude to set the amplitude to unity (retaining phase information), and then
    transforming back to the time domain via an inverse FFT.

    Args:
        pairs: List of tuples of raw signals (each tuple contains two numpy arrays).
        eps: A small constant to avoid division by zero in the normalization.

    Returns:
        A new list of tuples with spectrally whitened signals.
    """
    return [
        (_whiten(s1, eps), _whiten(s2, eps))
        for s1, s2 in tqdm(pairs, desc="Spectral whitening")
    ]


def one_bit_quantize_pairs(
    pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Quantize each signal in each pair to one bit.

    Args:
        pairs: List of tuples of raw signals (each tuple contains two numpy arrays).

    Returns:
        A new list of tuples with quantized signals.
    """
    quantized_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    sig1: np.ndarray
    sig2: np.ndarray

    for sig1, sig2 in tqdm(pairs, desc="One-bit quantization"):
        quantized_sig1 = np.where(sig1 >= 0, 1.0, -1.0)
        quantized_sig2 = np.where(sig2 >= 0, 1.0, -1.0)
        quantized_pairs.append((quantized_sig1, quantized_sig2))

    return quantized_pairs


def compute_cross_correlation(
    pairs: List[Tuple[np.ndarray, np.ndarray]], one_bit_quantization: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the average cross-correlation of a list of signal pairs after spectral whitening and optional one-bit
    quantization.

    This function processes a collection of signal pairs in several steps:
      1. Spectrally whitens each pair using `spectral_whiten_pairs`.
      2. Optionally applies one-bit quantization to the whitened pairs if `one_bit_quantization` is True.
      3. Computes the cross-correlation for each pair in the frequency domain:
         - For each pair, computes the real FFT of both signals.
         - Normalizes the FFT coefficients to unit magnitude (replacing any zeros with 1 to avoid division by zero),
           effectively converting them to phase-only representations.
         - Multiplies the FFT of the second signal with the conjugated, normalized FFT of the first signal.
         - Applies the inverse real FFT to obtain the cross-correlation in the time domain.
      4. Averages the individual cross-correlations over all pairs to produce a single representative cross-correlation.

    Parameters:
        pairs: A list where each element is a tuple containing two NumPy arrays, representing a pair of signals.
        one_bit_quantization: If True, applies one-bit quantization to the whitened signal pairs before computing their
            cross-correlation.

    Returns:
        An array representing the average cross-correlation computed across all processed signal pairs.
        An array representing the standard deviation of the cross-correlations across all pairs.
    """
    white_pairs: List[Tuple[np.ndarray, np.ndarray]] = spectral_whiten_pairs(
        pairs=pairs
    )

    if one_bit_quantization:
        white_pairs = one_bit_quantize_pairs(pairs=white_pairs)

    cc: List[np.ndarray] = []
    i: int

    for i in tqdm(range(len(white_pairs)), desc="Cross correlations"):
        a = np.fft.rfft(white_pairs[i][1])
        b = np.fft.rfft(white_pairs[i][0]).conj()
        # Normalize while replacing zeros with 1
        a = np.where(np.abs(a) == 0, 0, a / np.abs(a))
        b = np.where(np.abs(b) == 0, 0, b / np.abs(b))
        cc.append(np.fft.irfft(a * b))

    cc_arr = np.array(cc)
    cc_mean: np.ndarray = cc_arr.mean(axis=0)
    cc_std: np.ndarray = cc_arr.std(axis=0)

    return cc_mean, cc_std
