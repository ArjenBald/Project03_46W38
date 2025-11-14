# plots.py
# ------------------------------------------------------------
# Plotting utilities for wind resource assessment:
#   - wind speed histogram vs. fitted Weibull PDF
#   - wind rose (polar histogram of wind directions)
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Plot wind speed histogram and fitted Weibull PDF, return path to saved PNG
def plot_ws_hist_weibull(
    ws: pd.Series,
    k: float,
    A: float,
    height_label: str,
    start_year: int,
    end_year: int,
    lat: float,
    lon: float,
    output_dir: str | Path,
) -> Path:
    ws_arr = ws.dropna().to_numpy()
    x_max = max(np.quantile(ws_arr, 0.995), ws_arr.max())
    x = np.linspace(0.0, x_max, 400)
    pdf = weibull_min.pdf(x, c=k, loc=0.0, scale=A)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(ws_arr, bins=50, density=True, alpha=0.6, color="gray", edgecolor="none")
    ax.plot(x, pdf, "r-", linewidth=2)

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density [-]")
    ax.set_title(f"Wind speed distribution @ {height_label} ({start_year}-{end_year})")

    output_dir = Path(output_dir)
    plot_name = f"ws_hist_vs_weibull_{height_label}_{lat:.4f}_{lon:.4f}_{start_year}-{end_year}.png"
    plot_path = output_dir / plot_name
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path

# Plot wind rose (directional frequency) from wd series [deg FROM], return path to PNG
def plot_wind_rose(
    wd: pd.Series,
    height_label: str,
    start_year: int,
    end_year: int,
    lat: float,
    lon: float,
    output_dir: str | Path,
) -> Path:
    wd_arr = wd.dropna().to_numpy()

    n_sectors = 16
    edges_deg = np.linspace(0.0, 360.0, n_sectors + 1)
    counts, _ = np.histogram(wd_arr, bins=edges_deg)
    freqs = counts / counts.sum()

    theta_edges = np.deg2rad(edges_deg)
    widths = np.diff(theta_edges)
    theta_centers = theta_edges[:-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.2, 6.2))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(
        theta_centers,
        freqs,
        width=widths,
        align="edge",
        color="gray",
        edgecolor="none",
        alpha=0.8,
    )

    ax.set_title(f"Wind rose @ {height_label} ({start_year}-{end_year})")
    rticks = ax.get_yticks()
    ax.set_yticklabels([f"{t * 100:.0f}%" for t in rticks])

    output_dir = Path(output_dir)
    rose_name = f"wind_rose_{height_label}_{lat:.4f}_{lon:.4f}_{start_year}-{end_year}.png"
    rose_path = output_dir / rose_name
    fig.savefig(rose_path, dpi=150)
    plt.close(fig)
    return rose_path