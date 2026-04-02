import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
import torch
import os


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(
        filepath,
        sheet_name="Sheet1",
        skiprows=6
    )
    df.drop(columns=["PX_BID"], inplace=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['log_return'] = np.log(df["PX_LAST"] / df["PX_LAST"].shift(1))
    df.dropna(inplace=True)
    return df


def estimate_lambert_delta(y: np.ndarray) -> float:
    """
    Estimate the heavy-tail parameter δ of the Lambert W × Gaussian model
    via quasi maximum likelihood (Wiese et al. 2020, Section 5.3).

    The Lambert W × Gaussian model is:
        Y = X * exp(δ/2 * X²),   X ~ N(0, 1)

    The quasi-MLE minimises the negative log-likelihood:
        ℓ(δ) = -Σ [ log p_X(W_δ(y_i)) + log |dW_δ/dy|(y_i) ]

    where W_δ(y) = sign(y) * sqrt(Re(W(δ*y²)) / δ) is the back-transform
    from Y-space to X-space.
    """
    def neg_log_likelihood(delta: float) -> float:
        if delta <= 0:
            return 1e10
        w_vals = np.real(lambertw(delta * y**2))
        # Numerical guard: W must be finite and non-negative
        if np.any(~np.isfinite(w_vals)) or np.any(w_vals < 0):
            return 1e10
        x = np.sign(y) * np.sqrt(w_vals / delta)          # back-transformed X
        # log-derivative: d/dy W_δ(y) = 1 / (1 + W(δy²)) (chain rule simplification)
        log_det = -0.5 * np.log(1.0 + w_vals)
        # log p_X(x) under N(0,1): -0.5*x² - 0.5*log(2π)
        log_px = -0.5 * x**2
        return -np.sum(log_px + log_det)

    result = minimize_scalar(neg_log_likelihood, bounds=(1e-6, 5.0), method="bounded")
    return float(result.x)


def inverse_lambert_w(y: np.ndarray, delta: float) -> np.ndarray:
    """
    Apply the inverse Lambert W transform (Gauss → heavy-tail removal).

    Given standardised log-returns y (approximately Lambert W × Gaussian),
    recover the latent Gaussian variable:
        x = sign(y) * sqrt(Re(W(δ * y²)) / δ)
    """
    x = np.sign(y) * np.sqrt(np.real(lambertw(delta * y**2)) / delta)
    return x


def rolling_window(series: torch.Tensor, window_size: int = 127) -> torch.Tensor:
    if series.dim() == 1:
        series = series.unsqueeze(0)
    n_assets, T = series.shape
    windows = torch.stack([
        series[:, i:i + window_size]
        for i in range(T - window_size + 1)
    ])
    return windows


def preprocess(filepath: str) -> tuple[torch.Tensor, pd.DataFrame]:
    """
    Preprocessing pipeline (Figure 9, Wiese et al. 2020):
        1. Compute log-returns
        2. Normalise (zero mean, unit variance)
        3. Estimate Lambert W δ via quasi-MLE on the normalised returns
        4. Apply inverse Lambert W transform (remove heavy tails)
        5. Normalise again
        6. Extract rolling windows of length 127
    """
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    df = load_data(filepath)
    df = compute_log_returns(df)

    # Step 2: first normalisation
    df['log_return_normalized'] = scaler1.fit_transform(df[['log_return']])

    # Step 3: estimate δ from the normalised series
    y = df['log_return_normalized'].values
    delta = estimate_lambert_delta(y)
    print(f"Estimated Lambert W δ = {delta:.6f}")

    # Step 4: inverse Lambert W — map heavy-tailed → Gaussian
    df['log_return_transformed'] = inverse_lambert_w(y, delta)

    # Step 5: second normalisation (separate scaler, fresh fit)
    df['log_return_transformed_normalized'] = scaler2.fit_transform(
        df[['log_return_transformed']]
    )

    series = torch.tensor(
        df['log_return_transformed_normalized'].values, dtype=torch.float32
    )
    windows = rolling_window(series)

    return windows, df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    windows, df = preprocess("C01_Comdty(1).xlsx")

    print(df.head())
    print(f"\nNaN + Inf count:\n{df.isna().sum() + np.isinf(df.select_dtypes(include=np.number)).sum()}")
    print(f"\nWindows shape: {windows.shape}")
    print("Preprocessing completed successfully.")
