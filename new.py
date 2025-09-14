import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Path to folder containing .nc files
DATA_DIR = "./"   # change this to your folder path
nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))

# SQLite engine
engine = create_engine("sqlite:///argo.db")

def convert_time(val):
    """Convert Argo JULD to datetime (days since 1950-01-01)."""
    try:
        if np.issubdtype(type(val), np.number):
            return pd.to_datetime("1950-01-01") + pd.to_timedelta(float(val), unit="D")
        else:
            return pd.to_datetime(val)
    except Exception:
        return pd.NaT

all_dfs = []  # to keep track of everything for summary

for file in nc_files:
    print(f"📂 Processing {os.path.basename(file)} ...")
    ds = xr.open_dataset(file, decode_times=False, engine="netcdf4")

    # Extract variables
    float_id = ds.PLATFORM_NUMBER.values.astype(str)[0]
    lat = ds.LATITUDE.values
    lon = ds.LONGITUDE.values
    time = ds.JULD.values
    pressure = ds.PRES.values
    temperature = ds.TEMP.values
    salinity = ds.PSAL.values

    records = []

    for i in range(pressure.shape[0]):      # loop over profiles
        for j in range(pressure.shape[1]):  # loop over levels
            t_val = temperature[i, j]
            s_val = salinity[i, j]
            p_val = pressure[i, j]

            if pd.notna(t_val) and pd.notna(s_val) and pd.notna(p_val):
                records.append({
                    "float_id": float_id,
                    "profile_number": int(i),
                    "time": convert_time(time[i]),
                    "lat": float(lat[i]),
                    "lon": float(lon[i]),
                    "depth": float(p_val),
                    "temperature": float(t_val),
                    "salinity": float(s_val)
                })

    df = pd.DataFrame(records)

    if not df.empty:
        df.to_sql("argo_profiles", engine, if_exists="append", index=False)
        all_dfs.append(df)
        print(f"✅ Inserted {len(df)} rows from {os.path.basename(file)}")
    else:
        print(f"⚠️ No valid data in {os.path.basename(file)}")

print("🎉 All files processed and stored in argo.db")

# ==========================
# Quick summary
# ==========================
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    summary = {
        "unique_floats": merged["float_id"].nunique(),
        "total_profiles": merged["profile_number"].nunique(),
        "min_date": merged["time"].min(),
        "max_date": merged["time"].max(),
        "lat_min": merged["lat"].min(),
        "lat_max": merged["lat"].max(),
        "lon_min": merged["lon"].min(),
        "lon_max": merged["lon"].max()
    }

    print("\n📊 Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
else:
    print("⚠️ No data collected. Summary unavailable.")
