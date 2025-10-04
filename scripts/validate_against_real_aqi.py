#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Tuple, Dict
import time

import requests

City = Tuple[str, float, float]

CITIES: List[City] = [
    ("Mexico City", 19.4326, -99.1332),
    ("New York", 40.7128, -74.0060),
    ("London", 51.5074, -0.1278),
    ("Tokyo", 35.6895, 139.6917),
    ("Delhi", 28.6139, 77.2090),
    ("Cairo", 30.0444, 31.2357),
    ("Sydney", -33.8688, 151.2093),
    ("Lagos", 6.5244, 3.3792),
    ("Sao Paulo", -23.5505, -46.6333),
    ("Beijing", 39.9042, 116.4074),
]

# Map numeric AQI to qualitative category to compare fairly
# EPA US AQI bands
AQI_LEVELS = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]

def aqi_level(aqi: float) -> str:
    for lo, hi, name in AQI_LEVELS:
        if lo <= aqi <= hi:
            return name
    return "Unknown"


def fetch_model_aqi(lat: float, lon: float, base_url: str) -> Dict:
    r = requests.post(f"{base_url}/predict", json={"lat": lat, "lon": lon}, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_real_aqi_openmeteo(lat: float, lon: float) -> Dict:
    # Open-Meteo Air Quality API provides us_aqi hourly
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    us_aqi = hourly.get("us_aqi", [])
    if not times or not us_aqi:
        return {"status": "no-data"}
    # Take the last non-null value (closest to now with data)
    real_val = None
    for v in reversed(us_aqi):
        if v is not None:
            real_val = v
            break
    if real_val is None:
        return {"status": "no-data"}
    return {"status": "ok", "us_aqi": real_val}


def main():
    parser = argparse.ArgumentParser(description="Validate model AQI vs real Open-Meteo US AQI")
    parser.add_argument("--host", default="http://localhost", help="Model host base URL (default: http://localhost)")
    parser.add_argument("--port", type=int, default=5001, help="Model port (default: 5001)")
    parser.add_argument("--endpoint", default="/predict", help="Prediction endpoint path (default: /predict)")
    parser.add_argument("--cities", nargs="*", help="Optional list of 'Name:lat:lon' to override default cities")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests seconds (default: 0.2)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print per-city output")
    args = parser.parse_args()

    base = f"{args.host}:{args.port}".rstrip("/")
    url = f"{base}{args.endpoint}"  # for display only

    if args.cities:
        custom: List[City] = []
        for c in args.cities:
            try:
                name, slat, slon = c.split(":")
                custom.append((name, float(slat), float(slon)))
            except Exception:
                print(f"Invalid --cities item: {c} (expected Name:lat:lon)")
        cities = custom
    else:
        cities = CITIES

    print(f"Validating model {url} against Open-Meteo us_aqi ...\n")

    diffs = []
    level_matches = 0

    for name, lat, lon in cities:
        try:
            m = fetch_model_aqi(lat, lon, base)
            r = fetch_real_aqi_openmeteo(lat, lon)
            if r.get("status") != "ok":
                print(f"{name:12s} -> real AQI: no-data; model AQI: {m.get('air_quality_index')} ({m.get('air_quality_level')})")
                continue
            model_aqi = m.get("air_quality_index")
            model_lvl = m.get("air_quality_level")
            real_aqi = r.get("us_aqi")
            real_lvl = aqi_level(real_aqi)
            diff = abs(float(model_aqi) - float(real_aqi))
            diffs.append(diff)
            if model_lvl == real_lvl:
                level_matches += 1
            if args.pretty:
                print(json.dumps({
                    "city": name,
                    "coords": [lat, lon],
                    "model": {"aqi": model_aqi, "level": model_lvl},
                    "real": {"aqi": real_aqi, "level": real_lvl}
                }, ensure_ascii=False))
            else:
                print(f"{name:12s} -> model {model_aqi:3d} ({model_lvl}) | real {int(real_aqi):3d} ({real_lvl}) | Δ={diff:.1f}")
        except Exception as e:
            print(f"{name:12s} -> error: {e}")
        time.sleep(args.sleep)

    if diffs:
        mae = sum(diffs) / len(diffs)
        print(f"\nSummary: N={len(diffs)}  MAE={mae:.1f}  Level agreement={(level_matches/len(diffs))*100:.1f}%")
    else:
        print("\nSummary: No comparable data points obtained.")

if __name__ == "__main__":
    main()
