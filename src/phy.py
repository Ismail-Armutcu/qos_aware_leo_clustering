# src/phy.py
import numpy as np

C = 299_792_458.0


def fspl_db(range_m: np.ndarray, freq_hz: float) -> np.ndarray:
    # FSPL(dB) = 20log10(4*pi*d*f/c)
    return 20.0 * np.log10(4.0 * np.pi * range_m * freq_hz / C + 1e-12)


def gain_db_gaussian(theta_rad: np.ndarray, theta_3db_rad: float) -> np.ndarray:
    """
    Smooth mainlobe model:
    -3 dB at theta_3db, roll-off Gaussian-like.
    """
    a = np.log(2.0)
    g_lin = np.exp(-a * (theta_rad / (theta_3db_rad + 1e-12)) ** 2)
    return 10.0 * np.log10(g_lin + 1e-12)


def snr_lin(
    eirp_dbw: float,
    g_db: np.ndarray,
    fspl_dbv: np.ndarray,
    loss_db: float,
    noise_psd_dbw_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    # Pr(dBW) = EIRP + G - FSPL - losses
    pr_dbw = eirp_dbw + g_db - fspl_dbv - loss_db

    # N(dBW) = N0(dBW/Hz) + 10log10(B)
    n_dbw = noise_psd_dbw_hz + 10.0 * np.log10(bandwidth_hz + 1e-12)

    snr_db = pr_dbw - n_dbw
    return 10.0 ** (snr_db / 10.0)


def shannon_rate_mbps(snr_lin_v: np.ndarray, bandwidth_hz: float, eta: float) -> np.ndarray:
    rate_bps = eta * bandwidth_hz * np.log2(1.0 + snr_lin_v)
    return rate_bps / 1e6
