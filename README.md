## NASA Hackathon – Earth & Air Quality Explorer

Aplicación full‑stack para explorar la Tierra con mapas (Astro/Leaflet) y un backend de IA (Flask/PyTorch) que predice el Índice de Calidad del Aire (AQI) y entrega probabilidad de precipitación en tiempo real.

### Componentes
- Frontend: Astro + Leaflet, componentes en `src/components` y páginas en `src/pages`.
- Backend IA: Flask en `scripts/dsa_inference_api.py`.
- Data/ML: scripts en `scripts/` para descargar/armar dataset, preparar secuencias, entrenar y validar.

### Flujo de datos (alto nivel)
1) `scripts/build_openmeteo_us_aqi_dataset.py` descarga meteorología (Open‑Meteo Archive) y calidad del aire (Open‑Meteo Air Quality us_aqi) para ciudades globales, y guarda `processed_data/dataset_final_ml.csv`.
2) `scripts/prepare_dsa_sequences.py` genera ventanas temporales (24 horas) y guarda `X_dsa.npy`, `y_dsa.npy` y `feature_cols.txt`.
3) `scripts/train_dsa_model.py` entrena un LSTM (PyTorch) y guarda `processed_data/dsa_model.pt` y `processed_data/scaler.json`.
4) API `/predict` (Flask) consume el modelo, aplica el scaler y (opcional) calibración (`processed_data/aqi_calibration.json`), y llama a Open‑Meteo Forecast para probabilidad de precipitación.

---

## Cómo correr

### Frontend
1. Instalar dependencias Node
	- `npm install`
2. Desarrollo
	- `npm run dev` (http://localhost:4321)

### Backend IA
1. Arrancar API
	- `python3 scripts/dsa_inference_api.py` (http://localhost:5001)
2. Probar rápida
	- `python3 scripts/smoke_test_inference.py --pretty`

### Pipeline de datos/entrenamiento
1. Construir dataset real
	- `python3 scripts/build_openmeteo_us_aqi_dataset.py --days 21`
2. Preparar secuencias
	- `python3 scripts/prepare_dsa_sequences.py`
3. Entrenar modelo
	- `python3 scripts/train_dsa_model.py`
4. Validar vs real
	- `python3 scripts/validate_against_real_aqi.py`
5. Calibrar (opcional)
	- `python3 scripts/calculate_aqi_calibration.py`

---

## Endpoints
- POST `/predict` (body: `{ lat, lon }`)
  - Respuesta principal: `air_quality_index` (crudo) y `air_quality_index_calibrated` + niveles.
  - Extras: `precipitation_probability_now`, `precipitation_probability_next_hour` y metadatos de la consulta.

---

## Validación actual
- MAE ≈ 22.6 vs Open‑Meteo us_aqi (10 ciudades), con 30% de coincidencia por nivel.
- Mejoras rápidas sugeridas: ampliar días/ciudades, agregar gases/PMs como features, y recalibración post‑reentreno.

---

## Estructura del proyecto
```
/public          Archivos estáticos y tiles
/src             Frontend (Astro/Leaflet)
/scripts         Backend IA y pipeline de datos/ML
/processed_data  Artefactos del pipeline (dataset/modelo/scaler/calibración)
```

---

## Créditos
- Datos meteorológicos y de calidad del aire: Open‑Meteo
- Imágenes/satélite: NASA/USGS/ESA (capas en el frontend)

---

## Documentación
- Documento técnico del modelo: `docu.md`
- Guía de IA (ampliada): `docs/AI.md`
- Referencia de API: `docs/API.md`
