import pandas as pd
import os
import glob
import h5py
import numpy as np
from datetime import datetime

# Configuración de paths
METEO_CSV = "data/meteorologia_openmeteo.csv"
SAT_DATA_DIR = "nasa_data_functional/"
OUTPUT_CSV = "processed_data/dataset_final_ml.csv"

# 1. Cargar datos meteorológicos
meteo = pd.read_csv(METEO_CSV)
meteo["time"] = pd.to_datetime(meteo["time"])

# 2. Extraer variables clave de archivos HDF satelitales (ejemplo: MODIS AOD)
sat_rows = []

hdf_files = glob.glob(os.path.join(SAT_DATA_DIR, "*.HDF5"))
print(f"Archivos HDF5 encontrados en {SAT_DATA_DIR}: {len(hdf_files)}")
for fname in hdf_files:
    print(f"  - {fname}")

for hdf_file in hdf_files:
    try:
        with h5py.File(hdf_file, "r") as f:
            try:
                print(f"Intentando leer /Grid/precipitation en {fname} ...")
                if "/Grid/precipitation" in f:
                    precip = f["/Grid/precipitation"][:]
                    print(f"  Leído con éxito. Shape: {precip.shape}")
                    precip_flat = precip.flatten()
                    valid = precip_flat[np.isfinite(precip_flat)]
                    valid_nonneg = valid[valid >= 0]
                    precip_mean = valid_nonneg.mean() if valid_nonneg.size > 0 else None
                    print(f"Archivo: {fname} | min: {valid.min() if valid.size > 0 else 'NA'} | max: {valid.max() if valid.size > 0 else 'NA'} | válidos>=0: {valid_nonneg.size} | media: {precip_mean}")
                    print(f"  Primeros 10 valores (sin filtrar): {precip_flat[:10]}")
                else:
                    print(f"  La variable /Grid/precipitation NO existe en {fname}")
                    precip_mean = None
            except Exception as e:
                print(f"Error leyendo precipitación en {fname}: {e}")
                precip_mean = None
            fname = os.path.basename(hdf_file)
            date = None
            try:
                import re
                match = re.search(r"(\d{8})", fname)
                if match:
                    date_str = match.group(1)
                    date = datetime.strptime(date_str, "%Y%m%d")
                else:
                    print(f"No se pudo extraer fecha de: {fname}")
            except Exception as e:
                print(f"Error extrayendo fecha de {fname}: {e}")
            print(f"Archivo: {fname} | Fecha extraída: {date} | Precipitación media: {precip_mean}")
            sat_rows.append({"date": date, "precipitation": precip_mean})
    except (OSError, IOError):
        print(f"Archivo no válido o corrupto (omitido): {hdf_file}")


sat_df = pd.DataFrame(sat_rows)
# 3. Unir por fecha (puedes mejorar la lógica para matching horario si tienes timestamps más precisos)
if not sat_df.empty and "date" in sat_df.columns:
    sat_df = sat_df.dropna(subset=["date"]) # Solo fechas válidas
    # Convertir ambas columnas 'date' a datetime.date
    meteo["date"] = pd.to_datetime(meteo["time"]).dt.date
    sat_df["date"] = sat_df["date"].apply(lambda d: d.date() if hasattr(d, 'date') else d)
    final_df = pd.merge(meteo, sat_df, left_on="date", right_on="date", how="left")
else:
    print("No se extrajeron fechas válidas de los archivos satelitales. Se guardará solo el dataset meteorológico.")
    meteo["date"] = pd.to_datetime(meteo["time"]).dt.date
    final_df = meteo.copy()

# 4. Guardar dataset final
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset final guardado en {OUTPUT_CSV}")
