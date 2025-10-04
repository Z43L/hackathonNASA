import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math

def create_massive_geographic_dataset():
    """Crear dataset masivo con alta variabilidad geográfica para México"""
    
    # 50+ ubicaciones diversas en México con características geográficas únicas
    location_types = ["metropolitan","industrial","urban","coastal","mountain","border","tourist","agricultural","rural"]
    locations = [
        # ZONA METROPOLITANA (Alta contaminación)
        {"name": "Ciudad de México Centro", "lat": 19.4326, "lon": -99.1332, "altitude": 2240, "type": "metropolitan", "coast_distance": 300, "base_aqi": 110, "variance": 30},
        {"name": "Ecatepec", "lat": 19.6019, "lon": -99.0608, "altitude": 2240, "type": "metropolitan", "coast_distance": 300, "base_aqi": 125, "variance": 35},
        {"name": "Nezahualcóyotl", "lat": 19.4003, "lon": -98.9960, "altitude": 2240, "type": "metropolitan", "coast_distance": 300, "base_aqi": 120, "variance": 30},
        {"name": "Tlalnepantla", "lat": 19.5398, "lon": -99.1953, "altitude": 2280, "type": "metropolitan", "coast_distance": 300, "base_aqi": 115, "variance": 28},
        
        # CIUDADES INDUSTRIALES (Contaminación industrial)
        {"name": "Monterrey Centro", "lat": 25.6866, "lon": -100.3161, "altitude": 538, "type": "industrial", "coast_distance": 200, "base_aqi": 95, "variance": 25},
        {"name": "San Nicolás", "lat": 25.7417, "lon": -100.2958, "altitude": 500, "type": "industrial", "coast_distance": 200, "base_aqi": 105, "variance": 30},
        {"name": "Guadalupe NL", "lat": 25.6767, "lon": -100.2558, "altitude": 520, "type": "industrial", "coast_distance": 200, "base_aqi": 100, "variance": 28},
        {"name": "Tampico", "lat": 22.2331, "lon": -97.8614, "altitude": 10, "type": "industrial", "coast_distance": 5, "base_aqi": 85, "variance": 22},
        {"name": "Coatzacoalcos", "lat": 18.1340, "lon": -94.4520, "altitude": 10, "type": "industrial", "coast_distance": 15, "base_aqi": 90, "variance": 25},
        {"name": "Salamanca", "lat": 20.5737, "lon": -101.1954, "altitude": 1721, "type": "industrial", "coast_distance": 250, "base_aqi": 88, "variance": 23},
        
        # CIUDADES GRANDES (Contaminación urbana moderada)
        {"name": "Guadalajara Centro", "lat": 20.6597, "lon": -103.3496, "altitude": 1566, "type": "urban", "coast_distance": 200, "base_aqi": 75, "variance": 20},
        {"name": "Zapopan", "lat": 20.7206, "lon": -103.3844, "altitude": 1580, "type": "urban", "coast_distance": 200, "base_aqi": 70, "variance": 18},
        {"name": "Puebla Centro", "lat": 19.0414, "lon": -98.2063, "altitude": 2135, "type": "urban", "coast_distance": 150, "base_aqi": 65, "variance": 18},
        {"name": "Toluca", "lat": 19.2827, "lon": -99.6557, "altitude": 2667, "type": "urban", "coast_distance": 280, "base_aqi": 70, "variance": 20},
        {"name": "León", "lat": 21.1216, "lon": -101.6827, "altitude": 1815, "type": "urban", "coast_distance": 300, "base_aqi": 68, "variance": 19},
        {"name": "Querétaro", "lat": 20.5931, "lon": -100.3892, "altitude": 1820, "type": "urban", "coast_distance": 250, "base_aqi": 62, "variance": 17},
        {"name": "San Luis Potosí", "lat": 22.1565, "lon": -100.9855, "altitude": 1860, "type": "urban", "coast_distance": 280, "base_aqi": 64, "variance": 18},
        {"name": "Aguascalientes", "lat": 21.8853, "lon": -102.2916, "altitude": 1888, "type": "urban", "coast_distance": 320, "base_aqi": 60, "variance": 16},
        
        # CIUDADES COSTERAS (Viento marino limpia el aire)
        {"name": "Tijuana Centro", "lat": 32.5027, "lon": -117.0039, "altitude": 20, "type": "coastal", "coast_distance": 5, "base_aqi": 55, "variance": 15},
        {"name": "Mexicali", "lat": 32.6245, "lon": -115.4523, "altitude": 3, "type": "coastal", "coast_distance": 80, "base_aqi": 70, "variance": 20},
        {"name": "Ensenada", "lat": 31.8665, "lon": -116.5966, "altitude": 20, "type": "coastal", "coast_distance": 2, "base_aqi": 45, "variance": 12},
        {"name": "Mazatlán", "lat": 23.2494, "lon": -106.4103, "altitude": 3, "type": "coastal", "coast_distance": 1, "base_aqi": 50, "variance": 14},
        {"name": "Puerto Vallarta", "lat": 20.6534, "lon": -105.2253, "altitude": 7, "type": "coastal", "coast_distance": 1, "base_aqi": 48, "variance": 13},
        {"name": "Acapulco", "lat": 16.8531, "lon": -99.8237, "altitude": 3, "type": "coastal", "coast_distance": 1, "base_aqi": 52, "variance": 15},
        {"name": "Veracruz", "lat": 19.1738, "lon": -96.1342, "altitude": 10, "type": "coastal", "coast_distance": 2, "base_aqi": 58, "variance": 16},
        {"name": "Campeche", "lat": 19.8301, "lon": -90.5349, "altitude": 5, "type": "coastal", "coast_distance": 2, "base_aqi": 47, "variance": 13},
        {"name": "Cancún", "lat": 21.1619, "lon": -86.8515, "altitude": 10, "type": "coastal", "coast_distance": 1, "base_aqi": 44, "variance": 12},
        {"name": "Chetumal", "lat": 18.5001, "lon": -88.2960, "altitude": 10, "type": "coastal", "coast_distance": 15, "base_aqi": 42, "variance": 11},
        {"name": "La Paz BCS", "lat": 24.1426, "lon": -110.3128, "altitude": 20, "type": "coastal", "coast_distance": 2, "base_aqi": 40, "variance": 10},
        
        # CIUDADES MONTAÑOSAS (Altitud afecta condiciones)
        {"name": "Pachuca", "lat": 20.1011, "lon": -98.7591, "altitude": 2426, "type": "mountain", "coast_distance": 200, "base_aqi": 58, "variance": 16},
        {"name": "Xalapa", "lat": 19.5438, "lon": -96.9102, "altitude": 1427, "type": "mountain", "coast_distance": 60, "base_aqi": 54, "variance": 15},
        {"name": "Cuernavaca", "lat": 18.9267, "lon": -99.2233, "altitude": 1510, "type": "mountain", "coast_distance": 200, "base_aqi": 56, "variance": 16},
        {"name": "Morelia", "lat": 19.7006, "lon": -101.1944, "altitude": 1920, "type": "mountain", "coast_distance": 200, "base_aqi": 60, "variance": 17},
        {"name": "Zacatecas", "lat": 22.7709, "lon": -102.5832, "altitude": 2496, "type": "mountain", "coast_distance": 350, "base_aqi": 52, "variance": 14},
        {"name": "Guanajuato", "lat": 21.0190, "lon": -101.2574, "altitude": 2012, "type": "mountain", "coast_distance": 280, "base_aqi": 55, "variance": 15},
        
        # CIUDADES FRONTERIZAS (Factores transfronterizos)
        {"name": "Ciudad Juárez", "lat": 31.6904, "lon": -106.4245, "altitude": 1137, "type": "border", "coast_distance": 500, "base_aqi": 75, "variance": 22},
        {"name": "Nuevo Laredo", "lat": 27.4925, "lon": -99.5075, "altitude": 141, "type": "border", "coast_distance": 200, "base_aqi": 68, "variance": 19},
        {"name": "Reynosa", "lat": 26.0895, "lon": -98.2888, "altitude": 38, "type": "border", "coast_distance": 60, "base_aqi": 65, "variance": 18},
        {"name": "Matamoros", "lat": 25.8699, "lon": -97.5038, "altitude": 9, "type": "border", "coast_distance": 40, "base_aqi": 63, "variance": 17},
        
        # CIUDADES TURÍSTICAS (Regulaciones ambientales más estrictas)
        {"name": "Playa del Carmen", "lat": 20.6296, "lon": -87.0739, "altitude": 4, "type": "tourist", "coast_distance": 1, "base_aqi": 38, "variance": 10},
        {"name": "Cozumel", "lat": 20.4230, "lon": -86.9223, "altitude": 5, "type": "tourist", "coast_distance": 1, "base_aqi": 35, "variance": 9},
        {"name": "Huatulco", "lat": 15.7442, "lon": -96.2653, "altitude": 20, "type": "tourist", "coast_distance": 1, "base_aqi": 37, "variance": 10},
        {"name": "Los Cabos", "lat": 22.8905, "lon": -109.9167, "altitude": 6, "type": "tourist", "coast_distance": 1, "base_aqi": 36, "variance": 9},
        
        # CIUDADES AGRÍCOLAS (Contaminación por quemas y pesticidas)
        {"name": "Culiacán", "lat": 24.7999, "lon": -107.3841, "altitude": 54, "type": "agricultural", "coast_distance": 20, "base_aqi": 72, "variance": 20},
        {"name": "Hermosillo", "lat": 29.0729, "lon": -110.9559, "altitude": 237, "type": "agricultural", "coast_distance": 100, "base_aqi": 68, "variance": 19},
        {"name": "Villahermosa", "lat": 17.9892, "lon": -92.9476, "altitude": 10, "type": "agricultural", "coast_distance": 50, "base_aqi": 74, "variance": 21},
        {"name": "Tuxtla Gutiérrez", "lat": 16.7516, "lon": -93.1161, "altitude": 522, "type": "agricultural", "coast_distance": 80, "base_aqi": 66, "variance": 18},
        {"name": "Mérida", "lat": 20.9674, "lon": -89.5926, "altitude": 9, "type": "agricultural", "coast_distance": 35, "base_aqi": 62, "variance": 17},
        {"name": "Oaxaca", "lat": 17.0732, "lon": -96.7266, "altitude": 1555, "type": "agricultural", "coast_distance": 150, "base_aqi": 58, "variance": 16},
        
        # CIUDADES PEQUEÑAS/RURALES (Baja contaminación)
        {"name": "Colima", "lat": 19.2452, "lon": -103.7240, "altitude": 494, "type": "rural", "coast_distance": 40, "base_aqi": 45, "variance": 12},
        {"name": "Chilpancingo", "lat": 17.5506, "lon": -99.5024, "altitude": 1360, "type": "rural", "coast_distance": 120, "base_aqi": 48, "variance": 13},
        {"name": "Tlaxcala", "lat": 19.3139, "lon": -98.2404, "altitude": 2252, "type": "rural", "coast_distance": 180, "base_aqi": 50, "variance": 14},
        {"name": "Tepic", "lat": 21.5041, "lon": -104.8942, "altitude": 915, "type": "rural", "coast_distance": 50, "base_aqi": 46, "variance": 13}
    ]
    
    print(f"🌍 GENERANDO DATASET MASIVO GEOGRÁFICAMENTE DIVERSO")
    print(f"📍 {len(locations)} ubicaciones únicas en México")
    print(f"🏔️ Altitudes: {min(loc['altitude'] for loc in locations)}m - {max(loc['altitude'] for loc in locations)}m")
    print(f"🌊 Distancias costeras: {min(loc['coast_distance'] for loc in locations)}km - {max(loc['coast_distance'] for loc in locations)}km")
    print("=" * 80)
    
    all_rows = []
    start_date = datetime(2023, 1, 1)
    
    # 6 meses de datos (180 días) para capturar variaciones estacionales
    for location in locations:
        lat, lon = location["lat"], location["lon"]
        altitude = location["altitude"]
        location_type = location["type"]
        coast_distance = location["coast_distance"]
        base_aqi = location["base_aqi"]
        aqi_variance = location["variance"]
        
        print(f"Procesando {location['name']:25} | {location_type:12} | Alt: {altitude:4}m | AQI base: {base_aqi:3}")
        
        # Generar 180 días de datos cada 4 horas (6 mediciones/día)
        for day in range(180):
            for hour in [0, 4, 8, 12, 16, 20]:
                dt = start_date + timedelta(days=day, hours=hour)
                
                # === FEATURES GEOGRÁFICOS ÚNICOS ===
                
                # Factor de altitud (mayor altitud = menor presión, diferentes condiciones)
                altitude_factor = 1 - (altitude / 3000) * 0.2  # Reducir hasta 20% por altitud
                
                # Factor costero (proximidad al mar afecta humedad y viento)
                coastal_factor = max(0.5, 1 - (coast_distance / 500))  # Efecto hasta 500km
                
                # === PATRONES METEOROLÓGICOS ESPECÍFICOS POR REGIÓN ===
                
                # Temperatura base según ubicación y altitud
                if lat > 28:  # Norte árido
                    base_temp = 25 - (altitude / 300)
                elif lat > 23:  # Norte semiárido
                    base_temp = 23 - (altitude / 350)
                elif lat > 19:  # Centro templado
                    base_temp = 21 - (altitude / 400)
                else:  # Sur tropical
                    base_temp = 26 - (altitude / 250)
                
                # Variación diaria y estacional
                daily_temp_cycle = 12 * np.sin((hour - 6) * np.pi / 12)
                seasonal_temp = 8 * np.sin((day - 60) * 2 * np.pi / 365)  # Más calor en verano
                temperature = base_temp + daily_temp_cycle + seasonal_temp + np.random.normal(0, 3)
                
                # Humedad específica por región
                if location_type == "coastal":
                    base_humidity = 75 + coastal_factor * 15
                elif location_type == "mountain":
                    base_humidity = 60 - (altitude / 100)
                elif location_type == "agricultural":
                    base_humidity = 70
                else:
                    base_humidity = 65
                
                humidity = max(25, min(95, base_humidity - (temperature - 20) * 1.2 + np.random.normal(0, 8)))
                
                # Precipitación por región y estación
                is_dry_season = 300 <= day % 365 <= 120  # Nov-Abr (época seca)
                is_rainy_season = 120 < day % 365 < 300   # May-Oct (época de lluvias)
                
                if location_type == "coastal":
                    rain_prob = 0.15 if is_dry_season else 0.35
                elif location_type == "mountain":
                    rain_prob = 0.20 if is_dry_season else 0.45
                elif location_type == "agricultural":
                    rain_prob = 0.10 if is_dry_season else 0.40
                else:
                    rain_prob = 0.08 if is_dry_season else 0.25
                
                precipitation = np.random.exponential(2.5) if np.random.random() < rain_prob else 0
                
                # Viento específico por geografía
                if location_type == "coastal":
                    wind_speed = np.random.gamma(4, 2) * coastal_factor  # Más viento en costa
                elif location_type == "mountain":
                    wind_speed = np.random.gamma(2, 1.5) * altitude_factor  # Viento variable en montañas
                elif altitude > 2000:
                    wind_speed = np.random.gamma(3, 1.8)  # Más viento en altitud
                else:
                    wind_speed = np.random.gamma(2.5, 1.5)
                
                # Presión atmosférica por altitud
                pressure = 1013.25 * math.pow((1 - 0.0065 * altitude / 288.15), 5.255)
                pressure += np.random.normal(0, 8)
                
                # === CÁLCULO DE AQI CON ALTA VARIABILIDAD GEOGRÁFICA ===
                
                # AQI base específico de la ubicación
                current_aqi = base_aqi + np.random.normal(0, aqi_variance)
                
                # Factores por tipo de ubicación
                type_multipliers = {
                    "metropolitan": 1.3,
                    "industrial": 1.2,
                    "urban": 1.0,
                    "coastal": 0.7,
                    "mountain": 0.8,
                    "border": 1.1,
                    "tourist": 0.6,
                    "agricultural": 1.05,
                    "rural": 0.65
                }
                current_aqi *= type_multipliers.get(location_type, 1.0)
                
                # Factor de altitud (mayor altitud = menos oxígeno, más efectos de contaminación)
                if altitude > 2000:
                    current_aqi *= 1.15
                elif altitude > 1500:
                    current_aqi *= 1.08
                
                # Factor costero (viento marino limpia el aire)
                current_aqi *= (1 / coastal_factor)
                
                # Efectos meteorológicos mejorados
                if wind_speed < 2:
                    current_aqi *= 1.5
                elif wind_speed < 5:
                    current_aqi *= 1.25
                elif wind_speed > 12:
                    current_aqi *= 0.6
                
                if precipitation > 2:
                    current_aqi *= 0.5
                elif precipitation > 0.5:
                    current_aqi *= 0.75
                
                if humidity > 85:
                    current_aqi *= 1.2
                
                # Efectos de hora y estación
                if hour in [8, 20]:  # Rush hours
                    rush_multiplier = type_multipliers.get(location_type, 1.0) * 0.3 + 1.0
                    current_aqi *= rush_multiplier
                elif hour in [0, 4]:  # Madrugada
                    current_aqi *= 0.85
                
                if is_dry_season:
                    current_aqi *= 1.15
                
                # Limitaciones realistas por tipo
                if location_type == "tourist":
                    current_aqi = max(15, min(80, current_aqi))
                elif location_type == "metropolitan":
                    current_aqi = max(40, min(350, current_aqi))
                else:
                    current_aqi = max(15, min(300, current_aqi))
                
                # === DATOS SATELITALES CORRELACIONADOS ===
                
                # AOD correlacionado con AQI y geografía
                aod_base = 0.12 + (current_aqi / 200) * 0.6
                if location_type == "coastal":
                    aod_base *= 0.7
                elif location_type == "mountain":
                    aod_base *= 0.8
                elif location_type == "industrial":
                    aod_base *= 1.3
                
                aod = max(0.05, min(1.5, aod_base + np.random.normal(0, 0.1)))
                
                # NO2 troposférico
                no2_base = 8e-17 + (current_aqi / 180) * 3e-15
                if location_type in ["metropolitan", "industrial"]:
                    no2_base *= 1.4
                elif location_type in ["tourist", "rural"]:
                    no2_base *= 0.6
                
                no2 = max(1e-17, no2_base * (1 + np.random.normal(0, 0.4)))
                
                # Crear registro con features geográficos únicos
                row = {
                    'time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature_2m': round(temperature, 1),
                    'relative_humidity_2m': round(humidity),
                    'dew_point_2m': round(temperature - (100 - humidity) / 5, 1),
                    'apparent_temperature': round(temperature + np.random.uniform(-3, 3), 1),
                    'precipitation_x': round(precipitation, 2),
                    'windspeed_10m': round(wind_speed, 1),
                    'windgusts_10m': round(wind_speed * 1.2, 1),
                    'winddirection_10m': np.random.randint(0, 360),
                    'surface_pressure': round(pressure, 1),
                    'cloudcover': round(min(100, max(0, precipitation * 12 + humidity * 0.4 + np.random.normal(35, 20)))),
                    'date': dt.strftime('%Y-%m-%d'),
                    'precipitation_y': round(precipitation, 2),
                    'air_quality_index': round(current_aqi, 1),
                    'aod_satellite': round(aod, 4),
                    'no2_tropospheric': f"{no2:.2e}",
                    'lat': lat,
                    'lon': lon,
                    # NUEVOS FEATURES GEOGRÁFICOS ÚNICOS
                    'altitude': altitude,
                    'coast_distance': coast_distance,
                    'location_type': location_type
                }
                # one-hot de location_type
                for lt in location_types:
                    row[f'type_{lt}'] = 1 if location_type == lt else 0
                all_rows.append(row)
    
    # Crear DataFrame masivo
    df = pd.DataFrame(all_rows)
    
    # Crear directorio si no existe
    os.makedirs('processed_data', exist_ok=True)
    
    # Guardar dataset masivo
    df.to_csv('processed_data/dataset_final_ml.csv', index=False)
    
    print(f"\n🚀 DATASET MASIVO CREADO:")
    print(f"✅ {len(df):,} registros totales")
    print(f"✅ {len(locations)} ubicaciones geográficamente diversas")
    print(f"✅ 180 días de datos (6 mediciones/día)")
    print(f"✅ Features geográficos únicos: altitud, distancia_costa, tipo_ubicación")
    print(f"✅ Variabilidad AQI por ubicación garantizada")
    
    return df

if __name__ == "__main__":
    print("🔥 CREANDO DATASET MASIVO PARA MÁXIMA VARIABILIDAD GEOGRÁFICA")
    print("=" * 80)
    
    df = create_massive_geographic_dataset()
    
    print(f"\n📊 ESTADÍSTICAS DEL DATASET MASIVO:")
    print(f"- Registros totales: {len(df):,}")
    print(f"- Ubicaciones únicas: {df['lat'].nunique()}")
    print(f"- Rango temporal: {df['time'].min()} a {df['time'].max()}")
    print(f"- AQI mínimo: {df['air_quality_index'].min():.1f}")
    print(f"- AQI máximo: {df['air_quality_index'].max():.1f}")
    print(f"- AQI promedio general: {df['air_quality_index'].mean():.1f}")
    
    print(f"\n🌍 DIVERSIDAD GEOGRÁFICA:")
    print(f"- Altitudes: {df['altitude'].min()}m - {df['altitude'].max()}m")
    print(f"- Distancias costeras: {df['coast_distance'].min()}km - {df['coast_distance'].max()}km")
    print(f"- Tipos de ubicación: {df['location_type'].unique()}")
    
    print(f"\n🏙️ AQI PROMEDIO POR TIPO DE UBICACIÓN:")
    type_stats = df.groupby('location_type')['air_quality_index'].agg(['mean', 'std', 'count']).round(1)
    for location_type, stats in type_stats.iterrows():
        print(f"- {location_type:15}: {stats['mean']:6.1f} ± {stats['std']:5.1f} ({stats['count']:4} registros)")
    
    print(f"\n💾 Dataset guardado en: processed_data/dataset_final_ml.csv")
    print(f"🎯 ¡Variabilidad geográfica MAXIMIZADA!")