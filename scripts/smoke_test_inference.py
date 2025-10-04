#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Tuple

import requests

# A small curated set of cities across regions for quick sanity checks
CITIES: List[Tuple[str, float, float]] = [
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


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the AQI inference API.")
    parser.add_argument("--host", default="http://localhost", help="API host base URL (default: http://localhost)")
    parser.add_argument("--port", type=int, default=5001, help="API port (default: 5001)")
    parser.add_argument("--endpoint", default="/predict", help="Prediction endpoint path (default: /predict)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout seconds (default: 10.0)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON responses")
    args = parser.parse_args()

    base = f"{args.host}:{args.port}".rstrip("/")
    url = f"{base}{args.endpoint}"

    print(f"Hitting {url} ...\n")

    ok = True
    for name, lat, lon in CITIES:
        try:
            r = requests.post(url, json={"lat": lat, "lon": lon}, timeout=args.timeout)
            status = r.status_code
            print(f"{name:12s} -> HTTP {status}")
            if status != 200:
                ok = False
                print(f"  Error body: {r.text[:500]}")
                continue
            try:
                data = r.json()
            except Exception as e:
                ok = False
                print(f"  JSON parse error: {e}. Raw: {r.text[:500]}")
                continue

            if args.pretty:
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                aqi = data.get("air_quality_index")
                lvl = data.get("air_quality_level")
                used_lat = data.get("used_lat")
                used_lon = data.get("used_lon")
                print(f"  AQI: {aqi}  Level: {lvl}  used=({used_lat}, {used_lon})")
        except requests.RequestException as e:
            ok = False
            print(f"{name:12s} -> Request failed: {e}")

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
