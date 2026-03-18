import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
import torch
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

def normalize(df: pd.DataFrame, scaler: StandardScaler, col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = scaler.fit_transform(df[[col]])
    return df

def inverse_lambert_w(y: np.ndarray, delta: float = 0.1) -> np.ndarray:
    x = np.sign(y) * np.sqrt(np.real(lambertw(delta * y**2)) / delta)
    return x

def apply_inverse_lambert_w(df: pd.DataFrame) -> pd.DataFrame:
    df['log_return_transformed'] = inverse_lambert_w(df[['log_return_normalized']].values)
    return df

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
    scaler = StandardScaler()

    df = load_data(filepath)
    df = compute_log_returns(df)
    df = normalize(df, scaler, 'log_return', 'log_return_normalized')
    df = apply_inverse_lambert_w(df)
    df = normalize(df, scaler, 'log_return_transformed', 'log_return_transformed_normalized')

    series = torch.tensor(df['log_return_transformed_normalized'].values, dtype=torch.float32)
    windows = rolling_window(series)

    return windows, df

if __name__ == "__main__":
    windows, df = preprocess("C01_Comdty(1).xlsx")

    print(df.head())
    print(f"\nNaN + Inf count:\n{df.isna().sum() + np.isinf(df.select_dtypes(include=np.number)).sum()}")
    print(f"\nWindows shape: {windows.shape}")
    print("Preprocessing completed successfully.")