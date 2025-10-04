import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from train_dsa_model import DSAModel, WINDOW_SIZE
import math
import json
import requests
from datetime import datetime, timezone

# Cargar modelo y datos de referencia
MODEL_PATH = 'processed_data/dsa_model.pt'
DATA_PATH = 'processed_data/dataset_final_ml.csv'
SCALER_PATH = 'processed_data/scaler.json'
FEATURE_COLS_PATH = 'processed_data/feature_cols.txt'
CALIBRATION_PATH = 'processed_data/aqi_calibration.json'

app = Flask(__name__)

def get_aqi_level(aqi):
    """Convertir AQI numérico a nivel descriptivo"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def _apply_calibration(aqi_raw: float):
    """Apply linear calibration if available: y = slope*x + intercept + delta, clipped 0-500."""
    try:
        with open(CALIBRATION_PATH, 'r') as f:
            cal = json.load(f)
        p = cal.get('params', {})
        slope = float(p.get('slope', 1.0))
        intercept = float(p.get('intercept', 0.0))
        delta = float(p.get('delta', 0.0))
        y = slope * float(aqi_raw) + intercept + delta
        return max(0.0, min(500.0, y))
    except Exception:
        return float(aqi_raw)

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def _select_feature_cols(df: pd.DataFrame):
    """Select feature columns using saved feature order from training if available."""
    try:
        with open(FEATURE_COLS_PATH, 'r') as f:
            cols = [line.strip() for line in f if line.strip()]
        # Ensure all exist; if some missing, filter them out
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            raise ValueError("Empty feature cols from file")
        return cols
    except Exception:
        # Fallback to heuristic
        base = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
            'precipitation_x', 'windspeed_10m', 'windgusts_10m', 'winddirection_10m',
            'surface_pressure', 'cloudcover', 'precipitation_y', 'aod_satellite'
        ]
        extra = [c for c in ['altitude','coast_distance'] if c in df.columns]
        types = [c for c in df.columns if c.startswith('type_')]
        return base + extra + types

def _fetch_precip_probability(lat: float, lon: float):
    """Fetch precipitation probability (percent) for the current hour using Open-Meteo.

    Returns a tuple (prob_now, prob_next_hour, meta) where each prob is an int 0-100 or None.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": "precipitation_probability,precipitation",
            "forecast_days": 1,
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=4)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        probs = hourly.get("precipitation_probability", [])
        # Optional: precipitation intensity if needed later
        # precips = hourly.get("precipitation", [])

        if not times or not probs:
            return None, None, {"source": "open-meteo", "status": "no-hourly-data"}

        # Find index closest to current hour (handle naive/aware safely)
        def _to_naive(dt: datetime) -> datetime:
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        parsed = []
        for t in times:
            try:
                # Support both with and without timezone in the string
                dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                parsed.append(_to_naive(dt))
            except Exception:
                parsed.append(None)

        now_naive = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        # Choose the index with minimal absolute time difference to now
        best_idx = None
        best_diff = None
        for i, dt in enumerate(parsed):
            if dt is None:
                continue
            diff = abs((dt - now_naive).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i

        idx = best_idx if best_idx is not None else 0

        prob_now = probs[idx] if idx < len(probs) else None
        prob_next = probs[idx + 1] if (idx + 1) < len(probs) else None

        # Clamp and convert to int
        def _norm(x):
            if x is None:
                return None
            try:
                xi = int(round(float(x)))
                return max(0, min(100, xi))
            except Exception:
                return None

        return _norm(prob_now), _norm(prob_next), {"source": "open-meteo", "status": "ok"}
    except Exception as e:
        return None, None, {"source": "open-meteo", "status": "error", "message": str(e)}

def get_recent_sequence(df, lat, lon, window_size=WINDOW_SIZE):
    # Redondear coords para evitar problemas de precisión
    lat_q = round(float(lat), 4)
    lon_q = round(float(lon), 4)

    df_local = df.copy()
    df_local['lat_r'] = df_local['lat'].round(4)
    df_local['lon_r'] = df_local['lon'].round(4)

    # Intento 1: coincidencia exacta (redondeada)
    filtered = df_local[(df_local['lat_r'] == lat_q) & (df_local['lon_r'] == lon_q)]
    # Intento 2: si no alcanza, buscar vecino más cercano con suficientes datos
    used_lat, used_lon = None, None
    if len(filtered) < window_size:
        # candidatos únicos de sitios
        sites = df_local[['lat_r', 'lon_r']].drop_duplicates().rename(columns={'lat_r': 'lat', 'lon_r': 'lon'})
        if sites.empty:
            return None, None
        # calcular distancia y ordenar
        sites['dist_km'] = sites.apply(lambda r: _haversine_km(lat_q, lon_q, r['lat'], r['lon']), axis=1)
        sites = sites.sort_values('dist_km')
        # elegir el primer sitio con al menos window_size filas
        for _, row in sites.iterrows():
            cand = df_local[(df_local['lat_r'] == row['lat']) & (df_local['lon_r'] == row['lon'])]
            if len(cand) >= window_size:
                filtered = cand
                used_lat, used_lon = float(row['lat']), float(row['lon'])
                break
    else:
        used_lat, used_lon = lat_q, lon_q

    if len(filtered) < window_size:
        return None, None

    # Tomar la última ventana ordenada por tiempo
    seq = filtered.sort_values('time').iloc[-window_size:]
    feature_cols = _select_feature_cols(df_local)

    # Validación de columnas
    missing = [c for c in feature_cols if c not in seq.columns]
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing}")

    X = seq[feature_cols].astype(float).fillna(0).values.astype(np.float32)
    return X, (used_lat, used_lon)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon required'}), 400

    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    X, used_coords = get_recent_sequence(df, lat, lon)
    if X is None:
        return jsonify({'error': 'Not enough data for this location'}), 400

    # Aplicar el scaler guardado
    try:
        with open(SCALER_PATH, 'r') as f:
            scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        std = np.array(scaler['std'], dtype=np.float32)
        X = (X - mean) / std
    except Exception as e:
        # Si falta scaler, continuar sin normalizar
        pass

    X_tensor = torch.tensor(X).unsqueeze(0)  # batch=1
    # Determine input size from current features to avoid stale constants
    input_size = X.shape[1]
    try:
        with open(SCALER_PATH, 'r') as f:
            sc = json.load(f)
            nf = int(sc.get('n_features', input_size))
            input_size = nf
    except Exception:
        pass
    model = DSAModel(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).item()

    # Probabilidad de lluvia (tiempo real) desde Open-Meteo
    prob_now, prob_next, prob_meta = _fetch_precip_probability(lat, lon)

    # Calibración AQI
    aqi_raw = float(pred)
    aqi_cal = _apply_calibration(aqi_raw)

    # Respuesta
    return jsonify({
        'lat': lat,
        'lon': lon,
        'used_lat': used_coords[0] if used_coords else None,
        'used_lon': used_coords[1] if used_coords else None,
        'air_quality_index': round(aqi_raw),
        'air_quality_level': get_aqi_level(aqi_raw),
        'air_quality_index_calibrated': round(aqi_cal),
        'air_quality_level_calibrated': get_aqi_level(aqi_cal),
        'prediction_type': 'Air Quality Index (AQI)',
        'scale': '0-500 (0=Good, 500=Hazardous)',
        'precipitation_probability_now': prob_now,
        'precipitation_probability_next_hour': prob_next,
        'precipitation_probability_meta': prob_meta,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
