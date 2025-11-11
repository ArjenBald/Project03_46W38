import os
import pandas as pd
import numpy as np

# --- Test: Read turbine power-curve CSV files ----------------------------------
TURBINE_DATA_DIR = r"C:\Users\alexe\OneDrive\Documents\Education_DTU\GitHub\Project03_46W38\inputs\turbine_data"

TURBINES = {
    "NREL5":  {"hub_m": 90,  "rated_kW": 5000,  "curve_file": "NREL_Reference_5MW_126.csv"},
    "NREL15": {"hub_m": 150, "rated_kW": 15000, "curve_file": "2020ATB_NREL_Reference_15MW_240.csv"},
}


def load_power_curve_csv(path: str) -> dict[str, np.ndarray]:
    """Load power curve CSV (comma-separated, only first two columns: Wind Speed [m/s], Power [kW])."""
    df = pd.read_csv(path, sep=",", engine="python", header=0)
    df = df.iloc[:, :2].dropna()
    df.columns = ["Wind Speed [m/s]", "Power [kW]"]
    df = df.sort_values("Wind Speed [m/s]")

    ws = df["Wind Speed [m/s]"].to_numpy(dtype=float)
    pw = df["Power [kW]"].to_numpy(dtype=float)

    if pw[-1] > 0:
        ws = np.append(ws, ws[-1] + 0.01)
        pw = np.append(pw, 0.0)

    return {"ws": ws, "kW": pw}


if __name__ == "__main__":
    print("\n--- Testing turbine power-curve file loading ---")
    
    for key, t in TURBINES.items():
        curve_path = os.path.join(TURBINE_DATA_DIR, t["curve_file"])
        print(f"\nReading: {curve_path}")
        curve = load_power_curve_csv(curve_path)

        print(f"Turbine: {key}")
        print(f"Hub height: {t['hub_m']} m, Rated power: {t['rated_kW']} kW")
        print(f"Loaded {len(curve['ws'])} points")
        print("First 5 rows:")
        for v, p in list(zip(curve['ws'], curve['kW']))[:5]:
            print(f"  {v:>5.1f} m/s → {p:>8.2f} kW")
        print(f"... last point: {curve['ws'][-1]:.1f} m/s → {curve['kW'][-1]:.2f} kW")