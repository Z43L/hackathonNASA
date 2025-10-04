import pandas as pd
import numpy as np

def create_sequences(df, feature_cols, target_col, window_size=24, step=1):
    X, y = [], []
    for i in range(0, len(df) - window_size, step):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        y.append(df[target_col].iloc[i+window_size])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Cargar el dataset final
    df = pd.read_csv("processed_data/dataset_final_ml.csv", parse_dates=["time"])

    # Construir lista de features dinámica (básicos + geo + one-hot tipos)
    base_cols = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
        'precipitation_x', 'windspeed_10m', 'windgusts_10m', 'winddirection_10m',
        'surface_pressure', 'cloudcover', 'precipitation_y', 'aod_satellite'
    ]
    geo_cols = []
    if 'altitude' in df.columns: geo_cols.append('altitude')
    if 'coast_distance' in df.columns: geo_cols.append('coast_distance')
    # one-hot de location_type si existen
    type_cols = [c for c in df.columns if c.startswith('type_')]

    feature_cols = base_cols + geo_cols + type_cols
    print(f"📍 Usando {len(feature_cols)} features (base {len(base_cols)} + geo {len(geo_cols)} + tipo {len(type_cols)})")

    target_col = 'air_quality_index'
    print(f"🎯 Features: {len(feature_cols)} | Target: {target_col}")

    X, y = [], []
    for (lat, lon), group in df.groupby(['lat', 'lon']):
        group = group.sort_values('time')
        group[feature_cols] = group[feature_cols].fillna(0)
        group[target_col] = group[target_col].fillna(0)
        X_loc, y_loc = create_sequences(group, feature_cols, target_col, window_size=24, step=1)
        X.extend(X_loc)
        y.extend(y_loc)
    X = np.array(X)
    y = np.array(y)
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    np.save('processed_data/X_dsa.npy', X)
    np.save('processed_data/y_dsa.npy', y)
    # Guardar el orden de columnas de features para inferencia
    with open('processed_data/feature_cols.txt', 'w') as f:
        for c in feature_cols:
            f.write(c + '\n')
    print("Secuencias guardadas en processed_data/X_dsa.npy y y_dsa.npy")
