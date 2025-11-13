# analyzer.py
# ------------------------------------------------------------
# High-level class for wind resource assessment at a single site:
#   - load ERA5
#   - interpolate to site
#   - compute ws/wd time series
#   - optional power-law height
#   - Weibull fit + plots
#   - wind rose
#   - AEP for NREL 5 MW / 15 MW
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import xarray as xr

from .io import open_reanalysis_dataset
from .stats import wind_speed, wind_dir, power_law_extrapolate, fit_weibull
from .plots import plot_ws_hist_weibull, plot_wind_rose
from .turbines import TURBINES, load_power_curve_csv, power_at_ws


class WindResourceAnalyzer:
    """High-level interface for wind resource assessment at a single site."""

    # --- Initialization ---
    def __init__(
        self,
        reanalysis_dir: str | Path,
        turbine_data_dir: str | Path,
        output_dir: str | Path,
        lat: float,
        lon: float,
    ) -> None:
        self.reanalysis_dir = Path(reanalysis_dir)
        self.turbine_data_dir = Path(turbine_data_dir)
        self.output_dir = Path(output_dir)

        self.lat = float(lat)
        self.lon = float(lon)

        self.ds: Optional[xr.Dataset] = None
        self.df: Optional[pd.DataFrame] = None
        self.start_year: Optional[int] = None
        self.end_year: Optional[int] = None

        self.z: Optional[float] = None
        self.z_ref: Optional[float] = None
        self.alpha: Optional[float] = None

    # --- Load ERA5 reanalysis ---
    def load_reanalysis(self, start_year: int, end_year: int) -> None:
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.ds = open_reanalysis_dataset(
            reanalysis_dir=self.reanalysis_dir,
            start_year=self.start_year,
            end_year=self.end_year,
        )

    # --- Build interpolated time series at site ---
    def build_time_series(
        self,
        target_height: Optional[float] = None,
        alpha: float = 0.14,
    ) -> pd.DataFrame:
        if self.ds is None:
            raise RuntimeError("Dataset not loaded.")

        # Interpolation inside ERA5 box
        u100 = self.ds["u100"].interp(latitude=self.lat, longitude=self.lon)
        v100 = self.ds["v100"].interp(latitude=self.lat, longitude=self.lon)
        u10 = self.ds["u10"].interp(latitude=self.lat, longitude=self.lon)
        v10 = self.ds["v10"].interp(latitude=self.lat, longitude=self.lon)

        u100_s = u100.sortby("time").to_series().dropna()
        v100_s = v100.sortby("time").to_series().dropna()
        u10_s = u10.sortby("time").to_series().dropna()
        v10_s = v10.sortby("time").to_series().dropna()

        ws100 = wind_speed(u100_s, v100_s, name="ws100_ms")
        wd100 = wind_dir(u100_s, v100_s, name="wd100_deg")
        ws10 = wind_speed(u10_s, v10_s, name="ws10_ms")
        wd10 = wind_dir(u10_s, v10_s, name="wd10_deg")

        ws_z = None
        self.z = self.z_ref = self.alpha = None

        # Optional: compute wind speed at custom height z via power law
        if target_height is not None:
            z = float(target_height)
            if z <= 0:
                print("Non-positive target height; skipping custom height.")
            else:
                z_r = 100.0 if abs(z - 100.0) <= abs(z - 10.0) else 10.0
                self.z = z
                self.z_ref = z_r
                self.alpha = float(alpha)
                print(
                    f"Using power law: v(z)=v(z_r)*(z/z_r)^alpha, "
                    f"z={z:.1f} m, z_r={z_r:.1f} m, α={alpha:.3f}"
                )
                ws_ref = ws100 if z_r == 100 else ws10
                ws_z = power_law_extrapolate(
                    ws_ref, z_ref=z_r, z_target=z, alpha=alpha, name=f"ws{int(z)}m_ms"
                )

        df = (
            pd.concat([ws10, wd10, ws100, wd100, ws_z], axis=1)
            if ws_z is not None
            else pd.concat([ws10, wd10, ws100, wd100], axis=1)
        )

        self.df = df
        return df

    # --- Weibull fit ---
    def weibull_for_height(
        self,
        use_target_height_if_available: bool = True,
    ) -> Tuple[float, float, str]:
        if self.df is None:
            raise RuntimeError("Time series not available.")

        if use_target_height_if_available and self.z is not None:
            col = f"ws{int(self.z)}m_ms"
            height_label = f"{int(self.z)}m"
        else:
            col = "ws100_ms"
            height_label = "100m"

        ws = self.df[col].dropna()
        k, A = fit_weibull(ws)

        print(f"\nWeibull fit for {height_label}: k={k:.3f}, A={A:.3f}")
        return k, A, height_label

    # --- Save Weibull parameters CSV ---
    def save_weibull_summary(self, k: float, A: float, height_label: str) -> Path:
        if self.start_year is None or self.end_year is None:
            raise RuntimeError("Period unknown.")

        out = self.output_dir / (
            f"weibull_fit_{height_label}_{self.lat:.4f}_{self.lon:.4f}_"
            f"{self.start_year}-{self.end_year}.csv"
        )
        pd.DataFrame({"height": [height_label], "k_shape": [k], "A_scale": [A]}).to_csv(
            out, sep=";", index=False, encoding="utf-8-sig"
        )
        print("Saved Weibull summary:", out)
        return out

    # --- Weibull histogram plot ---
    def plot_weibull_distribution(self, k: float, A: float, height_label: str) -> Path:
        if self.df is None:
            raise RuntimeError("Time series not available.")

        col = (
            "ws100_ms"
            if height_label == "100m"
            else f"ws{int(height_label.replace('m', ''))}m_ms"
        )
        ws = self.df[col].dropna()

        out = plot_ws_hist_weibull(
            ws,
            k=k,
            A=A,
            height_label=height_label,
            start_year=self.start_year,
            end_year=self.end_year,
            lat=self.lat,
            lon=self.lon,
            output_dir=self.output_dir,
        )
        print("Saved wind speed plot:", out)
        return out

    # --- Wind rose ---
    def plot_wind_rose_for_height(self, height_label: str) -> Path:
        if self.df is None:
            raise RuntimeError("Time series not available.")

        wd = self.df["wd100_deg"].dropna()
        out = plot_wind_rose(
            wd,
            height_label=height_label,
            start_year=self.start_year,
            end_year=self.end_year,
            lat=self.lat,
            lon=self.lon,
            output_dir=self.output_dir,
        )
        print("Saved wind rose plot:", out)
        return out

    # --- Save time series CSV ---
    def save_time_series_csv(self) -> Path:
        if self.df is None:
            raise RuntimeError("Time series not available.")

        period = f"{self.start_year}-{self.end_year}"
        if self.z is not None:
            filename = f"ws_wd_10m_100m_{int(self.z)}m_{self.lat:.4f}_{self.lon:.4f}_{period}.csv"
        else:
            filename = f"ws_wd_10m_100m_{self.lat:.4f}_{self.lon:.4f}_{period}.csv"

        out = self.output_dir / filename
        self.df.to_csv(out, sep=";", index_label="time", encoding="utf-8-sig")
        print("Saved:", out)
        return out

    # --- Brief terminal summary ---
    def print_brief_summary(self) -> None:
        if self.df is None:
            raise RuntimeError("Time series not available.")

        print("\n=== Interpolated time series summary ===")
        print(f"Period: {self.df.index.min()} .. {self.df.index.max()}")
        msg = (
            f"Rows={len(self.df)}, "
            f"ws100 mean={self.df['ws100_ms'].mean():.3f}, "
            f"ws10 mean={self.df['ws10_ms'].mean():.3f}"
        )
        if self.z is not None:
            col = f"ws{int(self.z)}m_ms"
            if col in self.df:
                msg += f", ws{int(self.z)} mean={self.df[col].mean():.3f}"
        print(msg)

    # --- AEP calculation ---
    def compute_aep(
        self,
        turbine_key: str,
        year: int,
        alpha: float = 0.14,
    ) -> Tuple[float, Path]:
        if self.df is None:
            raise RuntimeError("Time series not available.")
        if turbine_key not in TURBINES:
            raise KeyError(f"Unknown turbine '{turbine_key}'.")

        turb = TURBINES[turbine_key]
        hub = float(turb["hub_m"])

        curve = load_power_curve_csv(self.turbine_data_dir / turb["curve_file"])

        year_slice = self.df.loc[f"{year}-01-01": f"{year}-12-31 23:00:00"]
        ws_ref = year_slice["ws100_ms"].dropna()

        ws_hub = power_law_extrapolate(ws_ref, z_ref=100.0, z_target=hub, alpha=alpha)
        pw = power_at_ws(ws_hub, curve)
        AEP_MWh = pw.sum() / 1000.0

        print(f"\nComputed AEP for {turbine_key} in {year}: {AEP_MWh:.2f} MWh")

        out = self.output_dir / f"aep_{turbine_key}_{year}_{self.lat:.4f}_{self.lon:.4f}.csv"
        pd.DataFrame(
            {"turbine": [turbine_key], "year": [year], "hub_height_m": [hub], "AEP_MWh": [AEP_MWh]}
        ).to_csv(out, sep=";", index=False, encoding="utf-8-sig")

        print("Saved AEP summary:", out)
        return AEP_MWh, out