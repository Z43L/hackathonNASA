#!/usr/bin/env python3
import argparse
import os
import json
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import requests

City = Tuple[str, float, float, str]  # name, lat, lon, type

# A balanced set of cities by region and climate
CITIES: List[City] = [
    ("Mexico City", 19.4326, -99.1332, "urban"),
    ("New York", 40.7128, -74.0060, "urban"),
    ("Los Angeles", 34.0522, -118.2437, "urban"),
    ("Sao Paulo", -23.5505, -46.6333, "urban"),
    ("Buenos Aires", -34.6037, -58.3816, "urban"),
    ("London", 51.5074, -0.1278, "urban"),
    ("Paris", 48.8566, 2.3522, "urban"),
    ("Madrid", 40.4168, -3.7038, "urban"),
    ("Berlin", 52.5200, 13.4050, "urban"),
    ("Rome", 41.9028, 12.4964, "urban"),
    ("Tokyo", 35.6895, 139.6917, "urban"),
    ("Seoul", 37.5665, 126.9780, "urban"),
    ("Beijing", 39.9042, 116.4074, "urban"),
    ("Singapore", 1.3521, 103.8198, "urban"),
    ("Delhi", 28.6139, 77.2090, "urban"),
    ("Cairo", 30.0444, 31.2357, "urban"),
    ("Riyadh", 24.7136, 46.6753, "urban"),
    ("Johannesburg", -26.2041, 28.0473, "urban"),
    ("Nairobi", -1.2921, 36.8219, "urban"),
    ("Sydney", -33.8688, 151.2093, "urban"),
    ("Melbourne", -37.8136, 144.9631, "urban"),
    ("Auckland", -36.8485, 174.7633, "urban"),
    ("Vancouver", 49.2827, -123.1207, "urban"),
    ("Montreal", 45.5019, -73.5674, "urban"),
]

HOURLY_WEATHER = [
    "temperature_2m","relative_humidity_2m","dew_point_2m","apparent_temperature",
    "precipitation","windspeed_10m","windgusts_10m","winddirection_10m","surface_pressure","cloudcover"
]

HOURLY_AIR = ["us_aqi", "pm10", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]


def fetch_openmeteo_block(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    # Weather archive
    w_url = "https://archive-api.open-meteo.com/v1/archive"
    w_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_WEATHER),
        "timezone": "UTC",
    }
    wr = requests.get(w_url, params=w_params, timeout=20)
    wr.raise_for_status()
    wj = wr.json()
    w_hourly = wj.get("hourly", {})
    w_df = pd.DataFrame(w_hourly)

    # Air quality (us_aqi)
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_AIR),
        "timezone": "UTC",
    }
    ar = requests.get(a_url, params=a_params, timeout=20)
    ar.raise_for_status()
    aj = ar.json()
    a_hourly = aj.get("hourly", {})
    a_df = pd.DataFrame(a_hourly)

    if w_df.empty:
        return pd.DataFrame()
    # Merge on time
    df = pd.merge(w_df, a_df, on="time", how="inner")
    return df


def build_dataset(days: int = 35, block_days: int = 7) -> pd.DataFrame:
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    rows = []
    for name, lat, lon, typ in CITIES:
        cur_start = start_dt
        while cur_start < end_dt:
            cur_end = min(cur_start + timedelta(days=block_days - 1), end_dt)
            s = cur_start.date().isoformat()
            e = cur_end.date().isoformat()
            try:
                df = fetch_openmeteo_block(lat, lon, s, e)
                if df.empty:
                    cur_start = cur_end + timedelta(days=1)
                    continue
                df["name"] = name
                df["lat"] = lat
                df["lon"] = lon
                # one-hot type
                df[f"type_{typ}"] = 1
                rows.append(df)
            except Exception:
                # Skip block on error
                pass
            cur_start = cur_end + timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Fill missing one-hot columns
    for t in sorted(set([c[5:] for c in out.columns if c.startswith("type_")])):
        col = f"type_{t}"
        if col not in out.columns:
            out[col] = 0
    # Normalize/rename columns to match training schema
    out.rename(columns={
        "precipitation": "precipitation_x",
        # keep aod_satellite absent (not available here), we'll fill 0
    }, inplace=True)

    # Required columns listing
    base_cols = [
        'temperature_2m','relative_humidity_2m','dew_point_2m','apparent_temperature',
        'precipitation_x','windspeed_10m','windgusts_10m','winddirection_10m',
        'surface_pressure','cloudcover'
    ]
    # precipitation_y and aod_satellite not present -> set to 0
    out['precipitation_y'] = 0.0
    out['aod_satellite'] = 0.0

    # Target
    out['air_quality_index'] = out['us_aqi']

    # Ensure time sorted per (lat, lon)
    out['time'] = pd.to_datetime(out['time'])
    out.sort_values(['lat','lon','time'], inplace=True)

    # Select final columns order (plus any type_*)
    type_cols = [c for c in out.columns if c.startswith('type_')]
    final_cols = ['time','name','lat','lon'] + base_cols + ['precipitation_y','aod_satellite','air_quality_index'] + type_cols
    out = out[final_cols]

    return out


def main():
    parser = argparse.ArgumentParser(description="Build dataset with Open-Meteo weather + us_aqi for training")
    parser.add_argument("--days", type=int, default=35, help="Days back to include (default: 35)")
    parser.add_argument("--block", type=int, default=7, help="Block size in days per request (default: 7)")
    parser.add_argument("--output", default="processed_data/dataset_final_ml.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Building real dataset from Open-Meteo (days={args.days})...")
    df = build_dataset(days=args.days, block_days=args.block)
    if df.empty:
        print("No data built.")
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")

if __name__ == "__main__":
    main()
