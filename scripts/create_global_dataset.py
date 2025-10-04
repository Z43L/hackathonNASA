import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math

# Utilidades climáticas sencillas por latitud/hemisferio

def lat_band(lat):
    a = abs(lat)
    if a < 10: return 'equatorial'
    if a < 23: return 'tropical'
    if a < 35: return 'subtropical'
    if a < 55: return 'temperate'
    if a < 66: return 'subpolar'
    return 'polar'


def estimate_base_temp(lat, altitude):
    # Base por latitud + efecto altitud (0.0065°C/m)
    bands = {
        'equatorial': 27,
        'tropical': 25,
        'subtropical': 22,
        'temperate': 15,
        'subpolar': 5,
        'polar': -5
    }
    b = bands[lat_band(lat)]
    return b - 0.0065 * altitude


def seasonal_offset(day_of_year, hemisphere):
    # Verano aprox día 172 (N), 355 (S) -> use seno con fase por hemisferio
    # amplitud mayor en latitudes fuera de trópicos
    phase = 172 if hemisphere == 'N' else 355
    return 8 * np.sin(2*np.pi*(day_of_year - phase)/365)


def wind_profile(location_type, coast_distance, altitude):
    if location_type == 'coastal':
        return np.random.gamma(4, 2) * max(0.6, 1 - coast_distance/500)
    if location_type == 'mountain':
        return np.random.gamma(2.5, 1.8) * (1 + min(0.5, altitude/3000))
    return np.random.gamma(2.2, 1.5)


def rain_probability(lat, location_type, region, month):
    band = lat_band(lat)
    # Monzones en sur/sureste asiático
    if region in ['South Asia', 'Southeast Asia']:
        return 0.45 if 5 <= month <= 10 else 0.15
    # Mediterráneo: lluvias en invierno
    if region in ['Southern Europe', 'North Africa']:
        return 0.40 if month in [11,12,1,2,3] else 0.15
    # Trópicos: convectivas frecuentes
    if band in ['equatorial','tropical']:
        return 0.35
    # Zonas templadas
    if band == 'temperate':
        return 0.25
    # Secas
    if location_type in ['desert','semiarid']:
        return 0.08
    return 0.18


def aqi_base_by_type(location_type):
    return {
        'megacity': 110,
        'metropolitan': 95,
        'industrial': 100,
        'urban': 70,
        'coastal': 45,
        'mountain': 55,
        'desert': 60,
        'semiarid': 65,
        'rural': 40
    }.get(location_type, 65)


def build_global_locations():
    # Lista de ciudades globales (lat, lon, alt(m), distancia costa(km), tipo, región, hemisferio, clima)
    # Nota: Altitudes aproximadas; suficientes para modelado sintético
    L = [
        # Américas
        ("New York", 40.7128, -74.0060, 10, 20, 'metropolitan', 'North America', 'N', 'temperate'),
        ("Los Angeles", 34.0522, -118.2437, 90, 20, 'coastal', 'North America', 'N', 'mediterranean'),
        ("Mexico City", 19.4326, -99.1332, 2240, 300, 'megacity', 'North America', 'N', 'highland'),
        ("Bogota", 4.7110, -74.0721, 2640, 600, 'mountain', 'South America', 'N', 'highland'),
        ("Lima", -12.0464, -77.0428, 154, 5, 'coastal', 'South America', 'S', 'desert'),
        ("Sao Paulo", -23.5505, -46.6333, 760, 70, 'industrial', 'South America', 'S', 'subtropical'),
        ("Buenos Aires", -34.6037, -58.3816, 25, 40, 'urban', 'South America', 'S', 'temperate'),
        ("Santiago", -33.4489, -70.6693, 520, 100, 'mountain', 'South America', 'S', 'mediterranean'),
        ("Toronto", 43.6511, -79.3470, 76, 30, 'urban', 'North America', 'N', 'continental'),
        ("Chicago", 41.8781, -87.6298, 180, 900, 'industrial', 'North America', 'N', 'continental'),
        # Europa
        ("London", 51.5074, -0.1278, 25, 60, 'urban', 'Western Europe', 'N', 'oceanic'),
        ("Paris", 48.8566, 2.3522, 35, 160, 'urban', 'Western Europe', 'N', 'oceanic'),
        ("Madrid", 40.4168, -3.7038, 667, 300, 'urban', 'Southern Europe', 'N', 'mediterranean'),
        ("Rome", 41.9028, 12.4964, 21, 25, 'urban', 'Southern Europe', 'N', 'mediterranean'),
        ("Berlin", 52.5200, 13.4050, 35, 200, 'urban', 'Central Europe', 'N', 'continental'),
        ("Moscow", 55.7558, 37.6173, 156, 640, 'industrial', 'Eastern Europe', 'N', 'continental'),
        ("Istanbul", 41.0082, 28.9784, 39, 5, 'megacity', 'Western Asia', 'N', 'temperate'),
        ("Stockholm", 59.3293, 18.0686, 28, 70, 'urban', 'Northern Europe', 'N', 'subpolar'),
        ("Oslo", 59.9139, 10.7522, 23, 10, 'urban', 'Northern Europe', 'N', 'subpolar'),
        # África
        ("Cairo", 30.0444, 31.2357, 23, 150, 'megacity', 'North Africa', 'N', 'desert'),
        ("Lagos", 6.5244, 3.3792, 2, 5, 'megacity', 'West Africa', 'N', 'tropical'),
        ("Nairobi", -1.2921, 36.8219, 1795, 480, 'mountain', 'East Africa', 'S', 'highland'),
        ("Johannesburg", -26.2041, 28.0473, 1753, 500, 'industrial', 'Southern Africa', 'S', 'subtropical'),
        ("Addis Ababa", 8.9806, 38.7578, 2355, 500, 'mountain', 'East Africa', 'N', 'highland'),
        ("Casablanca", 33.5731, -7.5898, 10, 5, 'coastal', 'North Africa', 'N', 'mediterranean'),
        # Asia
        ("Beijing", 39.9042, 116.4074, 43, 150, 'megacity', 'East Asia', 'N', 'continental'),
        ("Shanghai", 31.2304, 121.4737, 4, 5, 'megacity', 'East Asia', 'N', 'subtropical'),
        ("Tokyo", 35.6895, 139.6917, 40, 20, 'megacity', 'East Asia', 'N', 'temperate'),
        ("Seoul", 37.5665, 126.9780, 38, 40, 'industrial', 'East Asia', 'N', 'continental'),
        ("Delhi", 28.6139, 77.2090, 216, 900, 'megacity', 'South Asia', 'N', 'monsoon'),
        ("Mumbai", 19.0760, 72.8777, 14, 2, 'megacity', 'South Asia', 'N', 'monsoon'),
        ("Bangkok", 13.7563, 100.5018, 1, 30, 'urban', 'Southeast Asia', 'N', 'tropical'),
        ("Singapore", 1.3521, 103.8198, 15, 2, 'coastal', 'Southeast Asia', 'N', 'equatorial'),
        ("Jakarta", -6.2088, 106.8456, 8, 10, 'megacity', 'Southeast Asia', 'S', 'tropical'),
        ("Manila", 14.5995, 120.9842, 16, 5, 'coastal', 'Southeast Asia', 'N', 'tropical'),
        ("Riyadh", 24.7136, 46.6753, 610, 800, 'desert', 'Middle East', 'N', 'desert'),
        ("Tehran", 35.6892, 51.3890, 1200, 1100, 'industrial', 'Western Asia', 'N', 'continental'),
        # Oceanía
        ("Sydney", -33.8688, 151.2093, 58, 2, 'coastal', 'Oceania', 'S', 'temperate'),
        ("Melbourne", -37.8136, 144.9631, 31, 50, 'urban', 'Oceania', 'S', 'temperate'),
        ("Auckland", -36.8485, 174.7633, 30, 2, 'coastal', 'Oceania', 'S', 'oceanic'),
        ("Perth", -31.9505, 115.8605, 20, 15, 'coastal', 'Oceania', 'S', 'mediterranean'),
        # Polar/high
        ("Reykjavik", 64.1466, -21.9426, 46, 5, 'coastal', 'Northern Atlantic', 'N', 'subpolar'),
        ("Anchorage", 61.2181, -149.9003, 31, 30, 'coastal', 'North America', 'N', 'subpolar'),
        ("Ushuaia", -54.8019, -68.3030, 23, 5, 'coastal', 'Southern Atlantic', 'S', 'polar'),
    ]
    df = pd.DataFrame(L, columns=['name','lat','lon','altitude','coast_distance','location_type','region','hemisphere','climate'])
    return df


def create_global_dataset():
    cities = build_global_locations()
    print(f"🌐 Generando dataset global: {len(cities)} ciudades")

    start_date = datetime(2023,1,1)
    rows = []

    # one-hot templates
    location_types = sorted(cities['location_type'].unique().tolist())
    climates = sorted(cities['climate'].unique().tolist())
    regions = sorted(cities['region'].unique().tolist())

    for _, c in cities.iterrows():
        base_aqi = aqi_base_by_type(c['location_type'])
        for day in range(180):  # 6 meses
            for hour in [0,4,8,12,16,20]:
                dt = start_date + timedelta(days=day, hours=hour)
                month = dt.month
                doy = dt.timetuple().tm_yday

                # Temperatura base + estacional + ruido
                T0 = estimate_base_temp(c['lat'], c['altitude'])
                T_season = seasonal_offset(doy, c['hemisphere'])
                T_daily = 10 * np.sin((hour-6)*np.pi/12)
                temp = T0 + T_season + T_daily + np.random.normal(0, 2.5)

                # Humedad inversa a temperatura + efecto costa
                humidity = np.clip(75 - (temp-20)*1.2 + max(0, (200 - c['coast_distance']))*0.05 + np.random.normal(0,8), 20, 98)

                # Lluvia
                p_rain = rain_probability(c['lat'], c['location_type'], c['region'], month)
                precip = np.random.exponential(3) if np.random.rand() < p_rain else 0.0

                # Viento
                wind = wind_profile(c['location_type'], c['coast_distance'], c['altitude'])

                # Presión por altitud + ruido
                pressure = 1013.25 * math.pow((1 - 0.0065 * c['altitude'] / 288.15), 5.255) + np.random.normal(0,6)

                # Cloudcover correlacionado
                cloud = np.clip(precip*10 + humidity*0.3 + np.random.normal(30, 15), 0, 100)

                # AQI sintético calibrado global
                aqi = base_aqi + np.random.normal(0, 18)
                # Efectos meteo
                if wind < 2: aqi *= 1.35
                elif wind < 5: aqi *= 1.15
                elif wind > 12: aqi *= 0.75
                if precip > 2: aqi *= 0.55
                elif precip > 0.5: aqi *= 0.8
                if humidity > 85: aqi *= 1.10
                # Efectos horarias (tráfico)
                if hour in [8,20]: aqi *= 1.15
                elif hour in [0,4]: aqi *= 0.9
                # Límites por tipo
                if c['location_type'] in ['megacity','industrial','metropolitan']:
                    aqi = np.clip(aqi, 40, 350)
                elif c['location_type'] in ['coastal','rural']:
                    aqi = np.clip(aqi, 15, 120)
                else:
                    aqi = np.clip(aqi, 15, 300)

                # AOD y NO2 correlacionados
                aod = np.clip(0.1 + (aqi/220)*0.6 + np.random.normal(0,0.08), 0.05, 1.5)
                no2 = max(1e-17, (8e-17 + (aqi/180)*3e-15) * (1 + np.random.normal(0,0.35)))

                row = {
                    'time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'name': c['name'],
                    'temperature_2m': round(temp,1),
                    'relative_humidity_2m': int(round(humidity)),
                    'dew_point_2m': round(temp - max(0, (100-humidity))/5, 1),
                    'apparent_temperature': round(temp + np.random.uniform(-3,3), 1),
                    'precipitation_x': round(precip,2),
                    'windspeed_10m': round(wind,1),
                    'windgusts_10m': round(wind*1.3,1),
                    'winddirection_10m': np.random.randint(0,360),
                    'surface_pressure': round(pressure,1),
                    'cloudcover': int(round(cloud)),
                    'date': dt.strftime('%Y-%m-%d'),
                    'precipitation_y': round(precip,2),
                    'air_quality_index': round(float(aqi),1),
                    'aod_satellite': round(float(aod),4),
                    'no2_tropospheric': f"{no2:.2e}",
                    'lat': float(c['lat']),
                    'lon': float(c['lon']),
                    'altitude': float(c['altitude']),
                    'coast_distance': float(c['coast_distance']),
                    'location_type': c['location_type'],
                    'region': c['region'],
                    'hemisphere': c['hemisphere'],
                    'climate': c['climate']
                }
                # one-hot de location_type / climate / region
                for lt in location_types:
                    row[f'type_{lt}'] = 1 if c['location_type'] == lt else 0
                for cl in climates:
                    row[f'climate_{cl}'] = 1 if c['climate'] == cl else 0
                for rg in regions:
                    row[f'region_{rg}'] = 1 if c['region'] == rg else 0

                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs('processed_data', exist_ok=True)
    df.to_csv('processed_data/dataset_final_ml.csv', index=False)
    print(f"\n✅ Dataset global creado con {len(df):,} filas y {df[['lat','lon']].drop_duplicates().shape[0]} ciudades")
    return df

if __name__ == '__main__':
    df = create_global_dataset()
    try:
        print("\nMuestra:")
        print(df[['time','name','lat','lon','temperature_2m','precipitation_x','air_quality_index']].head().to_string(index=False))
    except Exception:
        print("\nMuestra no disponible (columnas faltantes)")
    print("Guardado en processed_data/dataset_final_ml.csv")
