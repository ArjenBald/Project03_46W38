# main.py
# ------------------------------------------------------------
# Purpose:
#   Compute an hourly wind-speed time series at 100 m (ws100) for one grid point
#   using u100/v100 from a single ERA5 NetCDF file, and save the result to CSV.
#   This is the first functional step toward calculating power output and AEP.
# ------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr

# Absolute path to one sample NetCDF file
NC_PATH = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\inputs\reanalysis_data\2000-2002.nc"

# Output directory 
OUTPUT_DIR = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\outputs\results"

# Site coordinates 
LAT, LON = 55.50, 7.75  # Alternatives: (55.50, 8.00), (55.75, 7.75), (55.75, 8.00)


def main():
    # Open a single NetCDF file (ERA5 hourly reanalysis)
    ds = xr.open_dataset(NC_PATH, engine="netcdf4", decode_timedelta=True)

    # Extract u100/v100 components at the specified grid point and order by time
    u = ds["u100"].sel(latitude=LAT, longitude=LON).sortby("time")
    v = ds["v100"].sel(latitude=LAT, longitude=LON).sortby("time")

    # Compute wind speed magnitude at 100 m [m/s]
    ws100 = np.sqrt(u.to_series() ** 2 + v.to_series() ** 2)
    ws100.name = "ws100_ms"

    # Print a brief statistical summary in the terminal
    print("=== WS@100m summary (single site, 2000–2002) ===")
    print(f"Site: lat={LAT:.2f}, lon={LON:.2f}")
    print(f"Length: {ws100.size} hours")
    print(f"Mean = {ws100.mean():.3f} m/s | Std = {ws100.std():.3f} m/s")
    print(f"Min  = {ws100.min():.3f} m/s | Max = {ws100.max():.3f} m/s")

    # Save the hourly time series to CSV (index = timestamp)
    out_csv = os.path.join(OUTPUT_DIR, f"ws100_{LAT:.2f}_{LON:.2f}_2000-2002.csv")
    ws100.to_csv(out_csv)
    print("Saved:", out_csv)

    # Note:
    # This script processes one ERA5 NetCDF file for a single location.


if __name__ == "__main__":
    main()