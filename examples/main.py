# main.py
# ------------------------------------------------------------
# Purpose:
#   For a user-specified location INSIDE the 2×2 ERA5 box:
#   (1) load multiple NetCDF files, (2) bilinearly interpolate u/v at 10 m and 100 m,
#   (3) compute wind speed and wind direction time series at both heights,
#   (4) allow user to specify start/end years for filtering,
#   (5) save a single CSV with ws/wd for 10 m and 100 m.
#   Direction is meteorological: degrees FROM which the wind blows [0..360).
# ------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Absolute path to the directory containing multiple ERA5 NetCDF files
REANALYSIS_DIR = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\inputs\reanalysis_data"

# Output directory 
OUTPUT_DIR = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\outputs\results"

# Target site INSIDE the box (Horns Rev 1 approx: 55°31′47″N, 7°54′22″E ≈ 55.53, 7.91)
LAT, LON = 55.53, 7.91


def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    """Wind speed magnitude [m/s] from u/v components."""
    s = np.sqrt(u.to_numpy()**2 + v.to_numpy()**2)
    return pd.Series(s, index=u.index, name="ws_ms")


def wind_dir(u: pd.Series, v: pd.Series) -> pd.Series:
    """Meteorological wind direction [deg FROM], 0=N, 90=E, using arctan2(-u, -v)."""
    wd_rad = np.arctan2(-u.to_numpy(), -v.to_numpy())
    wd_deg = (np.degrees(wd_rad) + 360.0) % 360.0
    return pd.Series(wd_deg, index=u.index, name="wd_deg")


def main():
    # Ask user for filtering years
    start_year = int(input("Enter start year (e.g., 1997): "))
    end_year = int(input("Enter end year (e.g., 2008): "))

    # Open multiple NetCDF files (ERA5 hourly reanalysis, combined by coordinates)
    files = tuple(sorted(str(p) for p in Path(REANALYSIS_DIR).glob("*.nc")))
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_timedelta=True,
        engine="netcdf4",
    )

    # Apply year filter to time dimension
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31 23:00:00"))

    # Bilinear interpolation to an arbitrary point inside the box (time preserved)
    u100 = ds["u100"].interp(latitude=LAT, longitude=LON)
    v100 = ds["v100"].interp(latitude=LAT, longitude=LON)
    u10  = ds["u10"].interp(latitude=LAT, longitude=LON)
    v10  = ds["v10"].interp(latitude=LAT, longitude=LON)

    # Convert to pandas Series and ensure chronological order
    u100_s = u100.sortby("time").to_series().dropna()
    v100_s = v100.sortby("time").to_series().dropna()
    u10_s  = u10.sortby("time").to_series().dropna()
    v10_s  = v10.sortby("time").to_series().dropna()

    # Compute speed and direction at both heights
    ws100 = wind_speed(u100_s, v100_s).rename("ws100_ms")
    wd100 = wind_dir(u100_s, v100_s).rename("wd100_deg")
    ws10  = wind_speed(u10_s,  v10_s ).rename("ws10_ms")
    wd10  = wind_dir(u10_s,  v10_s ).rename("wd10_deg")

    # Merge into a single DataFrame aligned by time
    df = pd.concat([ws10, wd10, ws100, wd100], axis=1)

    # Brief terminal summary
    print("\n=== Interpolated time series at 10 m and 100 m (inside-box point) ===")
    print(f"Period: {df.index.min()} .. {df.index.max()}")
    print(f"Rows: {len(df)} | ws100 mean={df['ws100_ms'].mean():.3f} m/s | ws10 mean={df['ws10_ms'].mean():.3f} m/s")

    # Save CSV
    period = f"{start_year}-{end_year}"
    out_csv = os.path.join(OUTPUT_DIR, f"ws_wd_10m_100m_{LAT:.4f}_{LON:.4f}_{period}.csv")
    df.to_csv(out_csv, sep=';', index_label='time', encoding='utf-8-sig')
    print("Saved:", out_csv)

    ds.close()


    # Note:
    # This covers the following minimal-spec items:
    #  - compute wind speed and direction from u/v,
    #  - compute ws/wd time series at 10 m and 100 m for a location inside the box (via interpolation),
    #  - load and combine multiple ERA5 NetCDF files into one continuous dataset,
    #  - apply user-specified start/end year filtering (inclusive),
    #  - export a single hourly CSV (Excel-friendly separator/encoding).

if __name__ == "__main__":
    main()