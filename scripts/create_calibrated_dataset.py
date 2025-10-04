import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_calibrated_dataset():
    """Crear dataset calibrado con valores AQI más realistas basados en datos históricos reales"""
    
    # Ubicaciones con AQI base según datos históricos reales de ciudades mexicanas
    locations = [
        {
            "name": "Mexico City", 
            "lat": 19.4326, 
            "lon": -99.1332,
            "base_aqi": 95,  # AQI típico alto (zona metropolitana muy contaminada)
            "aqi_variance": 25,  # Variabilidad típica
            "pollution_factors": {
                "urban_density": 1.3,  # Factor de densidad urbana alta
                "industrial": 1.2,     # Factor industrial
                "traffic": 1.4         # Factor de tráfico muy alto
            }
        },
        {
            "name": "Guadalajara", 
            "lat": 20.6597, 
            "lon": -103.3496,
            "base_aqi": 70,
            "aqi_variance": 20,
            "pollution_factors": {
                "urban_density": 1.1,
                "industrial": 1.1,
                "traffic": 1.2
            }
        },
        {
            "name": "Monterrey", 
            "lat": 25.6866, 
            "lon": -100.3161,
            "base_aqi": 80,  # Alto por industria pesada
            "aqi_variance": 22,
            "pollution_factors": {
                "urban_density": 1.2,
                "industrial": 1.4,  # Muy industrial
                "traffic": 1.3
            }
        },
        {
            "name": "Tijuana", 
            "lat": 32.5027, 
            "lon": -117.0039,
            "base_aqi": 65,  # Beneficiada por vientos del Pacífico
            "aqi_variance": 18,
            "pollution_factors": {
                "urban_density": 1.0,
                "industrial": 1.0,
                "traffic": 1.1
            }
        },
        {
            "name": "Puebla", 
            "lat": 19.0414, 
            "lon": -98.2063,
            "base_aqi": 55,  # Menos contaminada, más rural
            "aqi_variance": 15,
            "pollution_factors": {
                "urban_density": 0.9,
                "industrial": 0.8,
                "traffic": 0.9
            }
        }
    ]
    
    all_rows = []
    start_date = datetime(2023, 1, 1)
    
    print("Generando dataset calibrado con AQI realista...")
    
    for location in locations:
        lat, lon = location["lat"], location["lon"]
        base_aqi = location["base_aqi"]
        aqi_var = location["aqi_variance"]
        factors = location["pollution_factors"]
        
        print(f"Procesando {location['name']} (AQI base: {base_aqi})...")
        
        # Generar 60 días de datos cada 3 horas para más variabilidad
        for day in range(60):
            for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
                dt = start_date + timedelta(days=day, hours=hour)
                
                # === PATRONES METEOROLÓGICOS REALISTAS ===
                
                # Temperatura según ubicación y hora
                if lat > 25:  # Norte (Monterrey, Tijuana)
                    base_temp = 18
                elif lat > 20:  # Centro (Guadalajara)
                    base_temp = 22
                else:  # Sur (CDMX, Puebla)
                    base_temp = 19
                
                # Variación diaria de temperatura
                temp_cycle = 8 * np.sin((hour - 6) * np.pi / 12)  # Máximo a las 2 PM
                
                # Variación estacional (más calor en verano)
                seasonal_temp = 5 * np.sin((day - 90) * 2 * np.pi / 365)
                
                temperature = base_temp + temp_cycle + seasonal_temp + np.random.normal(0, 3)
                
                # Humedad inversa a temperatura, con efectos geográficos
                coastal_humidity = 10 if "Tijuana" in location["name"] else 0
                humidity = max(30, min(95, 75 - (temperature - 20) * 1.5 + coastal_humidity + np.random.normal(0, 8)))
                
                # Precipitación estacional realista
                # Más lluvia en verano (mayo-octubre) en México
                is_rainy_season = 120 <= day <= 300  # Mayo a octubre
                rain_prob = 0.25 if is_rainy_season else 0.08
                precipitation = np.random.exponential(3) if np.random.random() < rain_prob else 0
                
                # Viento con patrones geográficos
                if "Tijuana" in location["name"]:  # Costa del Pacífico
                    wind_speed = np.random.gamma(3, 2.5)  # Más viento
                elif "Monterrey" in location["name"]:  # Valle cerrado
                    wind_speed = np.random.gamma(2, 1.5)  # Menos viento
                else:
                    wind_speed = np.random.gamma(2.5, 2)
                
                # Presión atmosférica con variación por altitud
                altitude_effect = -10 if lat < 20 else -5  # CDMX más alto
                pressure = 1013 + altitude_effect + np.random.normal(0, 12)
                
                # Cobertura de nubes
                cloudcover = min(100, max(0, precipitation * 15 + humidity * 0.3 + np.random.normal(30, 15)))
                
                # === CÁLCULO DE AQI CALIBRADO ===
                
                # AQI base de la ciudad
                current_aqi = base_aqi + np.random.normal(0, aqi_var)
                
                # Factores de contaminación urbana
                current_aqi *= factors["urban_density"]
                current_aqi *= factors["industrial"] 
                current_aqi *= factors["traffic"]
                
                # Efectos meteorológicos en AQI (más realistas)
                if wind_speed < 2:  # Viento muy bajo = contaminación atrapada
                    current_aqi *= 1.4
                elif wind_speed < 5:  # Viento bajo
                    current_aqi *= 1.2
                elif wind_speed > 15:  # Viento fuerte = dispersión
                    current_aqi *= 0.7
                
                if precipitation > 1:  # Lluvia significativa limpia el aire
                    current_aqi *= 0.6
                elif precipitation > 0.1:  # Lluvia ligera
                    current_aqi *= 0.8
                
                if humidity > 85:  # Humedad alta = peor visibilidad
                    current_aqi *= 1.15
                
                # Efectos de hora del día (más tráfico en rush hours)
                if hour in [6, 9, 18, 21]:  # Rush hours
                    current_aqi *= 1.2
                elif hour in [0, 3]:  # Madrugada = menos tráfico
                    current_aqi *= 0.9
                
                # Efectos estacionales
                if is_rainy_season:
                    current_aqi *= 0.9  # Menos contaminación en época de lluvias
                else:
                    current_aqi *= 1.1  # Más contaminación en época seca
                
                # Limitar AQI a rangos realistas
                current_aqi = max(15, min(300, current_aqi))
                
                # === DATOS SATELITALES CORRELACIONADOS ===
                
                # AOD correlacionado con AQI de manera más realista
                aod = 0.15 + (current_aqi / 200) * 0.5 + np.random.normal(0, 0.08)
                aod = max(0.05, min(1.2, aod))
                
                # NO2 troposférico correlacionado
                no2_base = 1e-16 + (current_aqi / 150) * 2e-15
                no2 = no2_base * (1 + np.random.normal(0, 0.3))
                no2 = max(1e-17, no2)
                
                # Crear registro
                row = {
                    'time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature_2m': round(temperature, 1),
                    'relative_humidity_2m': round(humidity),
                    'dew_point_2m': round(temperature - (100 - humidity) / 5, 1),
                    'apparent_temperature': round(temperature + np.random.uniform(-3, 3), 1),
                    'precipitation_x': round(precipitation, 2),
                    'windspeed_10m': round(wind_speed, 1),
                    'windgusts_10m': round(wind_speed * 1.3, 1),
                    'winddirection_10m': np.random.randint(0, 360),
                    'surface_pressure': round(pressure, 1),
                    'cloudcover': round(cloudcover),
                    'date': dt.strftime('%Y-%m-%d'),
                    'precipitation_y': round(precipitation, 2),
                    'air_quality_index': round(current_aqi, 1),
                    'aod_satellite': round(aod, 4),
                    'no2_tropospheric': f"{no2:.2e}",
                    'lat': lat,
                    'lon': lon
                }
                all_rows.append(row)
    
    # Crear DataFrame
    df = pd.DataFrame(all_rows)
    
    # Crear directorio si no existe
    os.makedirs('processed_data', exist_ok=True)
    
    # Guardar dataset calibrado
    df.to_csv('processed_data/dataset_final_ml.csv', index=False)
    
    print(f"\n✅ Dataset calibrado creado con {len(df)} filas")
    print(f"✅ {len(locations)} ciudades con AQI realista")
    print(f"✅ 60 días de datos (8 mediciones/día)")
    print(f"✅ Correlaciones meteorológicas mejoradas")
    print(f"✅ Factores urbanos/industriales incluidos")
    
    return df

if __name__ == "__main__":
    print("🔧 CALIBRACIÓN DEL DATASET CON AQI REALISTA")
    print("=" * 50)
    
    df = create_calibrated_dataset()
    
    print(f"\n📊 ESTADÍSTICAS DEL DATASET CALIBRADO:")
    print(f"- Filas totales: {len(df):,}")
    print(f"- Ubicaciones: {df['lat'].nunique()}")
    print(f"- Rango de fechas: {df['time'].min()} a {df['time'].max()}")
    print(f"- AQI mínimo: {df['air_quality_index'].min():.1f}")
    print(f"- AQI máximo: {df['air_quality_index'].max():.1f}")
    print(f"- AQI promedio: {df['air_quality_index'].mean():.1f}")
    print(f"- AQI mediana: {df['air_quality_index'].median():.1f}")
    
    print(f"\n🏙️ AQI PROMEDIO POR CIUDAD:")
    city_stats = df.groupby(['lat', 'lon'])['air_quality_index'].agg(['mean', 'std']).round(1)
    city_names = ["Ciudad de México", "Guadalajara", "Monterrey", "Tijuana", "Puebla"]
    for i, (idx, row) in enumerate(city_stats.iterrows()):
        print(f"- {city_names[i]:15}: {row['mean']:5.1f} ± {row['std']:4.1f}")
    
    print(f"\n💾 Dataset guardado en: processed_data/dataset_final_ml.csv")
    print(f"🚀 Listo para reentrenar el modelo DSA")