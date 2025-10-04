import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Configuración: Cambia estos valores según tu región y fechas de interés
LAT = 40.4168  # Ejemplo: Madrid
LON = -3.7038
START_DATE = "2023-01-01"
END_DATE = "2023-01-10"
OUTPUT_CSV = "data/meteorologia_openmeteo.csv"

# Variables meteorológicas a descargar
VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "windspeed_10m", "windgusts_10m",
    "winddirection_10m", "surface_pressure", "cloudcover"
]

# Construir la URL de la API
base_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join(VARIABLES),
    "timezone": "auto"
}

print(f"Descargando datos meteorológicos de Open-Meteo para lat={LAT}, lon={LON}...")
response = requests.get(base_url, params=params)
response.raise_for_status()
data = response.json()

# Convertir a DataFrame
hourly = data["hourly"]
df = pd.DataFrame(hourly)

# Guardar CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Datos meteorológicos guardados en {OUTPUT_CSV}")
