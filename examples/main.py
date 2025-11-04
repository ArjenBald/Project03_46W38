# main_v0.py
# ------------------------------------------------------------
# Purpose:
#   Inspect one ERA5 NetCDF file to understand its internal structure.
#   This confirms that all input files (1997–2008) share the same format:
#   variables u10, v10, u100, v100 with dimensions (time, latitude, longitude).
# ------------------------------------------------------------

from __future__ import annotations
import xarray as xr

# Absolute path to one sample NetCDF file
NC_PATH = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\inputs\reanalysis_data\2000-2002.nc"

def main():
    # Open a single NetCDF file (ERA5 hourly reanalysis)
    ds = xr.open_dataset(NC_PATH, engine="netcdf4", decode_timedelta=True)

    # === 1. Dataset summary ===
    print("=== DATASET SUMMARY ===")
    print(ds)

    # === 2. Data variables (u/v at 10m and 100m) ===
    print("\n=== DATA VARIABLES ===")
    for name in ds.data_vars:
        var = ds[name]
        print(f"- {name}: shape={tuple(var.shape)}, dims={var.dims}")


    # Note:
    # All ERA5 input files in this project share the same structure,
    # only the time range (e.g., 1997–1999, 2000–2002, etc.) differs.
    # Therefore, inspecting one file is sufficient to understand the dataset.

if __name__ == "__main__":
    main()