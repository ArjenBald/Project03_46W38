# main.py
# ------------------------------------------------------------
# Purpose:
#   For a user-specified location INSIDE the 2×2 ERA5 box:
#   (1) load and combine multiple NetCDF files (ERA5 hourly reanalysis),
#   (2) bilinearly interpolate u/v at 10 m and 100 m for the given site,
#   (3) compute wind speed and wind direction time series at both heights,
#   (4) allow user to specify start/end years for filtering,
#   (5) optionally compute wind speed at a user-defined height z
#       using the power law profile (v = v(z_r) * (z/z_r)^α),
#   (6) save a single CSV with ws/wd at 10 m, 100 m, and optionally z m,
#   (7) fit Weibull distribution (k, A) for the selected height and
#       export a compact CSV summary of the fit,
#   (8) plot wind speed distribution (histogram vs. fitted Weibull curve)
#       for the selected location and height.
#   Direction is meteorological: degrees FROM which the wind blows [0..360).
# ------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

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

    # Ask user for target height z (optional power-law profile)
    z_in = input("\nEnter target height z in meters (press Enter to skip): ").strip()
    ws_z = None
    if z_in:
        z = float(z_in)
        if z <= 0:
            print("Non-positive z provided; skipping custom height.")
            z_in = ""
        else:
            # choose reference height closest to z; just store params for later
            z_r = 100 if abs(z - 100) <= abs(z - 10) else 10
            alpha = 0.14  # coastal default
            print(f"Using power law: v(z) = v(z_r) * (z/z_r)^alpha, z={z} m, z_r={z_r} m, α={alpha}")
    
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

    # Optionally compute wind speed at user-specified height z using power law
    if z_in:
        ws_ref = ws100 if z_r == 100 else ws10
        ws_z = (ws_ref * (z / z_r) ** alpha).rename(f"ws{int(z)}m_ms")

    # Merge into a single DataFrame aligned by time
    if ws_z is not None:
        df = pd.concat([ws10, wd10, ws100, wd100, ws_z], axis=1)
    else:
        df = pd.concat([ws10, wd10, ws100, wd100], axis=1)

    # --- Fit Weibull distribution for selected height ---
    if ws_z is not None:
        col = f"ws{int(z)}m_ms"
        height_label = f"{int(z)}m"
    else:
        col = "ws100_ms"
        height_label = "100m"

    ws_for_fit = df[col].dropna()
    k, loc, A = weibull_min.fit(ws_for_fit, floc=0.0)

    print(f"\nWeibull fit for {height_label}: k={k:.3f}, A={A:.3f}")

    # --- Plot: wind speed histogram vs fitted Weibull ---
    ws = ws_for_fit.to_numpy()
    x_max = max(np.quantile(ws, 0.995), ws.max())
    x = np.linspace(0, x_max, 400)
    pdf = weibull_min.pdf(x, c=k, loc=0.0, scale=A)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(ws, bins=50, density=True, alpha=0.6, color="gray", edgecolor="none")
    ax.plot(x, pdf, "r-", linewidth=2)

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density [-]")
    ax.set_title(f"Wind speed distribution @ {height_label} ({start_year}-{end_year})")

    plot_name = (
        f"ws_hist_vs_weibull_{height_label}_{LAT:.4f}_{LON:.4f}_{start_year}-{end_year}.png"
    )
    plot_path = os.path.join(OUTPUT_DIR, plot_name)
    fig.savefig(plot_path, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved wind speed distribution plot:", plot_path)

    # Brief terminal summary
    print("\n=== Interpolated time series at 10 m and 100 m (inside-box point) ===")
    print(f"Period: {df.index.min()} .. {df.index.max()}")
    msg = (
        f"Rows: {len(df)} | "
        f"ws100 mean={df['ws100_ms'].mean():.3f} m/s | "
        f"ws10 mean={df['ws10_ms'].mean():.3f} m/s"
    )
    if ws_z is not None:
        msg += f" | ws{int(z)} mean={df[f'ws{int(z)}m_ms'].mean():.3f} m/s"
    print(msg)

    # Save CSV
    period = f"{start_year}-{end_year}"

    # Add z-level info to filename if applicable
    if ws_z is not None:
        filename = f"ws_wd_10m_100m_{int(z)}m_{LAT:.4f}_{LON:.4f}_{period}.csv"
    else:
        filename = f"ws_wd_10m_100m_{LAT:.4f}_{LON:.4f}_{period}.csv"

    out_csv = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_csv, sep=';', index_label='time', encoding='utf-8-sig')
    print("Saved:", out_csv)

    # --- Export Weibull fit summary ---
    fit_summary = pd.DataFrame({
    "height": [height_label],
    "k_shape": [k],
    "A_scale": [A],
})

    fit_csv = os.path.join(
        OUTPUT_DIR,
        f"weibull_fit_{height_label}_{LAT:.4f}_{LON:.4f}_{start_year}-{end_year}.csv"
    )
    fit_summary.to_csv(fit_csv, sep=';', index=False, encoding='utf-8-sig')
    print("Saved Weibull summary:", fit_csv)

    ds.close()


    # Note:
    # This covers the following minimal-spec items:
    #  - compute wind speed and direction from u/v,
    #  - compute ws/wd time series at 10 m and 100 m for a location inside the box (via interpolation),
    #  - load and combine multiple ERA5 NetCDF files into one continuous dataset,
    #  - apply user-specified start/end year filtering (inclusive),
    #  - compute wind speed at a user-specified height z using the power law profile,
    #  - export a single hourly CSV (Excel-friendly separator/encoding, includes z-level in filename if provided),
    #  - fit and export Weibull distribution parameters (k, A) for selected height,
    #  - plot wind speed distribution (histogram vs. fitted Weibull curve) for the selected height and period.

if __name__ == "__main__":
    main()