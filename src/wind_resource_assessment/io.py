# io.py
# ------------------------------------------------------------
# ERA5 dataset loading utilities:
#   - collect multiple NetCDF files
#   - open them as a single dataset
#   - slice by time range
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import xarray as xr

# Load ERA5 reanalysis dataset from multiple NetCDF files and filter by years
def open_reanalysis_dataset(
    reanalysis_dir: str | Path,
    start_year: int,
    end_year: int,
) -> xr.Dataset:
    reanalysis_dir = Path(reanalysis_dir)
    files = sorted(reanalysis_dir.glob("*.nc"))

    ds = xr.open_mfdataset(
        [str(p) for p in files],
        combine="by_coords",
        decode_timedelta=True,
        engine="netcdf4",
    )

    # Filter dataset by selected time range
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31 23:00:00"))
    return ds