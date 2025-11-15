# turbines.py
# ------------------------------------------------------------
# Turbine utilities:
#   - NREL 5 MW / 15 MW metadata
#   - load turbine power curve from CSV
#   - interpolate power from wind speed
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# NREL reference turbines and their power-curve file names
TURBINES: dict[str, dict[str, object]] = {
    "NREL5": {
        "hub_m": 90.0,
        "rated_kW": 5000.0,
        "curve_file": "NREL_Reference_5MW_126.csv",
    },
    "NREL15": {
        "hub_m": 150.0,
        "rated_kW": 15000.0,
        "curve_file": "2020ATB_NREL_Reference_15MW_240.csv",
    },
}

# Load turbine power curve CSV (wind speed [m/s], power [kW])
def load_power_curve_csv(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    df = pd.read_csv(path, sep=",", engine="python", header=0)

    df = df.iloc[:, :2].dropna()
    df.columns = ["Wind Speed [m/s]", "Power [kW]"]
    df = df.sort_values("Wind Speed [m/s]")

    ws = df["Wind Speed [m/s]"].to_numpy(dtype=float)
    pw = df["Power [kW]"].to_numpy(dtype=float)

    # Ensure curve ends with zero power (cut-out)
    if pw[-1] > 0.0:
        ws = np.append(ws, ws[-1] + 0.01)
        pw = np.append(pw, 0.0)

    return {"ws": ws, "kW": pw}

# Interpolate turbine power [kW] for each wind speed value in series
def power_at_ws(ws_series: pd.Series, curve: dict[str, np.ndarray]) -> pd.Series:
    p = np.interp(
        ws_series.to_numpy(),
        curve["ws"],
        curve["kW"],
        left=0.0,
        right=0.0,
    )
    return pd.Series(p, index=ws_series.index, name="power_kW")