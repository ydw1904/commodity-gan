"""
Stylized facts metrics for evaluating synthetic financial time series.

Reference: Cont (2001) "Empirical properties of asset returns: stylized facts and
statistical implications", Quantitative Finance.
"""
import numpy as np
import torch
from statsmodels.stats.stattools import jarque_bera


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def autocorrelation(series: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Autocorrelation of a 1-D series at lags 1..max_lag."""
    series = series - series.mean()
    var = np.dot(series, series)
    acf = np.array([
        np.dot(series[lag:], series[:-lag]) / var if lag > 0 else 1.0
        for lag in range(max_lag + 1)
    ])
    return acf  # shape (max_lag+1,), acf[0]=1


def stylized_facts(
    real: torch.Tensor | np.ndarray,
    fake: torch.Tensor | np.ndarray,
    max_lag: int = 50,
) -> dict:
    """
    Compute stylized-facts metrics comparing real vs. synthetic return sequences.

    Both inputs should be flat 1-D arrays (or will be flattened).

    Returns a dict with:
        acf_returns_real / fake    — ACF of raw returns (should be ~0)
        acf_abs_returns_real/fake  — ACF of |returns| (volatility clustering)
        kurtosis_real / fake       — Excess kurtosis (fat tails; normal=0)
        jb_pvalue_real / fake      — Jarque–Bera p-value (normality test)
    """
    r = _to_numpy(real).flatten()
    f = _to_numpy(fake).flatten()

    acf_r = autocorrelation(r, max_lag)
    acf_f = autocorrelation(f, max_lag)
    acf_abs_r = autocorrelation(np.abs(r), max_lag)
    acf_abs_f = autocorrelation(np.abs(f), max_lag)

    from scipy.stats import kurtosis
    kurt_r = kurtosis(r, fisher=True)
    kurt_f = kurtosis(f, fisher=True)

    jb_r = jarque_bera(r)[1]   # p-value
    jb_f = jarque_bera(f)[1]

    return {
        "acf_returns_real": acf_r,
        "acf_returns_fake": acf_f,
        "acf_abs_returns_real": acf_abs_r,
        "acf_abs_returns_fake": acf_abs_f,
        "kurtosis_real": float(kurt_r),
        "kurtosis_fake": float(kurt_f),
        "jb_pvalue_real": float(jb_r),
        "jb_pvalue_fake": float(jb_f),
    }


def print_summary(metrics: dict) -> None:
    """Print a human-readable summary of stylized facts metrics."""
    print(f"{'Metric':<30} {'Real':>12} {'Fake':>12}")
    print("-" * 56)
    for key in ("kurtosis", "jb_pvalue"):
        print(f"{key:<30} {metrics[f'{key}_real']:>12.4f} {metrics[f'{key}_fake']:>12.4f}")
    print(f"\nACF of returns    (lags 1-5): real {metrics['acf_returns_real'][1:6].round(3)}  "
          f"fake {metrics['acf_returns_fake'][1:6].round(3)}")
    print(f"ACF of |returns|  (lags 1-5): real {metrics['acf_abs_returns_real'][1:6].round(3)}  "
          f"fake {metrics['acf_abs_returns_fake'][1:6].round(3)}")
