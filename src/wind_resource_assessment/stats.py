# stats.py
# ------------------------------------------------------------
# Wind statistics utilities:
#   - wind speed and direction from u/v
#   - power-law vertical extrapolation
#   - Weibull parameter fitting
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

# Compute wind speed magnitude [m/s] from u/v components
def wind_speed(u: pd.Series, v: pd.Series, name: str = "ws_ms") -> pd.Series:
    s = np.sqrt(u.to_numpy()**2 + v.to_numpy()**2)
    return pd.Series(s, index=u.index, name=name)

# Compute meteorological wind direction [deg FROM], 0=N, 90=E
def wind_dir(u: pd.Series, v: pd.Series, name: str = "wd_deg") -> pd.Series:
    wd_rad = np.arctan2(-u.to_numpy(), -v.to_numpy())
    wd_deg = (np.degrees(wd_rad) + 360.0) % 360.0
    return pd.Series(wd_deg, index=u.index, name=name)

# Extrapolate wind speed using power law: v(z) = v(z_ref) * (z/z_ref)^alpha
def power_law_extrapolate(
    ws_ref: pd.Series,
    z_ref: float,
    z_target: float,
    alpha: float = 0.14,
    name: str | None = None,
) -> pd.Series:
    if name is None:
        name = f"ws{int(z_target)}m_ms"
    factor = (z_target / z_ref) ** alpha
    return (ws_ref * factor).rename(name)

# Fit Weibull distribution (k shape, A scale) with fixed loc=0
def fit_weibull(ws: pd.Series) -> tuple[float, float]:
    data = ws.dropna().to_numpy()
    k, loc, A = weibull_min.fit(data, floc=0.0)
    return float(k), float(A)