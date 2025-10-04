import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuración
N_LOCATIONS = 20  # número de ubicaciones aleatorias
DAYS = 60        # días de datos
START_DATE = datetime(2023, 1, 1)

# Rango de México aproximado
LAT_RANGE = (14.5, 32.7)
LON_RANGE = (-117.1, -86.7)

# Variables meteorológicas y satelitales
columns = [
    'time','temperature_2m','relative_humidity_2m','dew_point_2m','apparent_temperature',
    'precipitation_x','windspeed_10m','windgusts_10m','winddirection_10m','surface_pressure',
    'cloudcover','date','precipitation_y','air_quality_index','lat','lon'
]

rows = []
for loc in range(N_LOCATIONS):
    lat = round(random.uniform(*LAT_RANGE), 4)
    lon = round(random.uniform(*LON_RANGE), 4)
    for day in range(DAYS):
        for hour in range(24):
            dt = START_DATE + timedelta(days=day, hours=hour)
            row = [
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                round(random.uniform(0, 35), 1),  # temperature_2m
                random.randint(10, 100),           # relative_humidity_2m
                round(random.uniform(-5, 25), 1),  # dew_point_2m
                round(random.uniform(-5, 40), 1),  # apparent_temperature
                round(random.uniform(0, 10), 1),   # precipitation_x
                round(random.uniform(0, 30), 1),   # windspeed_10m
                round(random.uniform(0, 50), 1),   # windgusts_10m
                random.randint(0, 360),            # winddirection_10m
                round(random.uniform(900, 1050),1),# surface_pressure
                random.randint(0, 100),            # cloudcover
                dt.strftime('%Y-%m-%d'),           # date
                round(random.uniform(0, 10), 2),   # precipitation_y
                random.randint(0, 500),            # air_quality_index (0-500 scale)
                lat,
                lon
            ]
            rows.append(row)

# Guardar CSV
out_df = pd.DataFrame(rows, columns=columns)
out_df.to_csv('processed_data/dataset_final_ml.csv', index=False)
print(f"Generado processed_data/dataset_final_ml.csv con {len(rows)} filas y {N_LOCATIONS} ubicaciones aleatorias.")
