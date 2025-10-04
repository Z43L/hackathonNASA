#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import requests
from datetime import datetime

City = Tuple[str, float, float]

# Curated global list (~40 cities)
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
    ("Los Angeles", 34.0522, -118.2437),
    ("Chicago", 41.8781, -87.6298),
    ("Toronto", 43.6532, -79.3832),
    ("Madrid", 40.4168, -3.7038),
    ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050),
    ("Rome", 41.9028, 12.4964),
    ("Istanbul", 41.0082, 28.9784),
    ("Moscow", 55.7558, 37.6176),
    ("Seoul", 37.5665, 126.9780),
    ("Bangkok", 13.7563, 100.5018),
    ("Jakarta", -6.2088, 106.8456),
    ("Singapore", 1.3521, 103.8198),
    ("Hong Kong", 22.3193, 114.1694),
    ("Shanghai", 31.2304, 121.4737),
    ("Shenzhen", 22.5431, 114.0579),
    ("Karachi", 24.8607, 67.0011),
    ("Tehran", 35.6892, 51.3890),
    ("Riyadh", 24.7136, 46.6753),
    ("Dubai", 25.2048, 55.2708),
    ("Johannesburg", -26.2041, 28.0473),
    ("Nairobi", -1.2921, 36.8219),
    ("Lima", -12.0464, -77.0428),
    ("Bogota", 4.7110, -74.0721),
    ("Buenos Aires", -34.6037, -58.3816),
    ("Santiago", -33.4489, -70.6693),
    ("Melbourne", -37.8136, 144.9631),
    ("Auckland", -36.8485, 174.7633),
    ("Vancouver", 49.2827, -123.1207),
    ("Montreal", 45.5019, -73.5674),
]

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


def fetch_model_aqi(base_url: str, lat: float, lon: float, timeout: float = 10.0) -> Optional[float]:
    try:
        r = requests.post(f"{base_url}/predict", json={"lat": lat, "lon": lon}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return float(data.get("air_quality_index"))
    except Exception:
        return None


def fetch_real_aqi_openmeteo(lat: float, lon: float, timeout: float = 10.0) -> Optional[float]:
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {"latitude": lat, "longitude": lon, "hourly": "us_aqi", "timezone": "auto"}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        us_aqi = hourly.get("us_aqi", [])
        if not us_aqi:
            return None
        # last non-null
        for v in reversed(us_aqi):
            if v is not None:
                return float(v)
        return None
    except Exception:
        return None

@dataclass
class CalibrationResult:
    slope: float
    intercept: float
    delta: float
    r2: float
    mae_raw: float
    mae_calibrated: float
    level_acc_raw: float
    level_acc_calibrated: float
    n: int


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # slope and intercept via least squares; also compute r^2
    x_mean, y_mean = x.mean(), y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return 1.0, 0.0, 0.0
    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    # r^2
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return slope, intercept, r2


def grid_search_delta(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    # Choose delta maximizing level agreement; tie-break on MAE
    best_delta = 0.0
    best_acc = -1.0
    best_mae = float("inf")
    for d in range(-50, 51):
        y_hat = slope * x + intercept + d
        y_hat = np.clip(y_hat, 0, 500)
        acc = (np.array([aqi_level(v) for v in y_hat]) == np.array([aqi_level(v) for v in y])).mean()
        mae = np.abs(y - y_hat).mean()
        if acc > best_acc or (acc == best_acc and mae < best_mae):
            best_acc = acc
            best_mae = mae
            best_delta = float(d)
    return best_delta


def calibrate(base_url: str, cities: List[City], sleep: float = 0.15) -> CalibrationResult:
    xs, ys = [], []
    for name, lat, lon in cities:
        m = fetch_model_aqi(base_url, lat, lon)
        r = fetch_real_aqi_openmeteo(lat, lon)
        if m is not None and r is not None:
            xs.append(m)
            ys.append(r)
        time.sleep(sleep)
    if not xs:
        raise RuntimeError("No calibration pairs collected")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    slope, intercept, r2 = fit_linear(x, y)
    delta = grid_search_delta(x, y, slope, intercept)

    y_hat_raw = x  # identity
    y_hat_cal = np.clip(slope * x + intercept + delta, 0, 500)

    mae_raw = float(np.abs(y - y_hat_raw).mean())
    mae_cal = float(np.abs(y - y_hat_cal).mean())
    acc_raw = float((np.array([aqi_level(v) for v in y_hat_raw]) == np.array([aqi_level(v) for v in y])).mean())
    acc_cal = float((np.array([aqi_level(v) for v in y_hat_cal]) == np.array([aqi_level(v) for v in y])).mean())

    return CalibrationResult(
        slope=float(slope),
        intercept=float(intercept),
        delta=float(delta),
        r2=float(r2),
        mae_raw=mae_raw,
        mae_calibrated=mae_cal,
        level_acc_raw=acc_raw,
        level_acc_calibrated=acc_cal,
        n=len(x),
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate AQI calibration parameters (linear + delta)")
    parser.add_argument("--host", default="http://localhost", help="Model host base URL (default: http://localhost)")
    parser.add_argument("--port", type=int, default=5001, help="Model port (default: 5001)")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests seconds")
    parser.add_argument("--output", default="processed_data/aqi_calibration.json", help="Output calibration JSON path")
    args = parser.parse_args()

    base = f"{args.host}:{args.port}".rstrip("/")

    print(f"Collecting calibration pairs from {len(CITIES)} cities...")
    res = calibrate(base, CITIES, sleep=args.sleep)

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "open-meteo us_aqi",
        "params": {"slope": res.slope, "intercept": res.intercept, "delta": res.delta},
        "metrics": {
            "n": res.n,
            "r2": res.r2,
            "mae_raw": res.mae_raw,
            "mae_calibrated": res.mae_calibrated,
            "level_acc_raw": res.level_acc_raw,
            "level_acc_calibrated": res.level_acc_calibrated,
        },
    }

    print("Calibration summary:")
    print(json.dumps(payload, indent=2))

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved calibration to {args.output}")

if __name__ == "__main__":
    main()
