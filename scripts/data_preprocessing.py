#!/usr/bin/env python3
"""
Script de Preprocesamiento de Datos para Predicción Meteorológica con Atención Dispersa
===================================================================================

Este script implementa el preprocesamiento de datos siguiendo la metodología inspirada
en DeepSeek's Sparse Attention para datos geoespaciales.

Pasos implementados:
1. Lectura de archivos HDF5/NetCDF
2. Extracción de variables clave (precipitationCal, temperatura, etc.)
3. Creación de "fotogramas" temporales (grillas 2D)
4. Aplanamiento de grillas en vectores 1D
5. Construcción de secuencias temporales
6. División en conjuntos de entrenamiento, validación y prueba
"""

import os
import glob
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class EarthDataPreprocessor:
    """
    Preprocesador de datos de la NASA para predicción meteorológica
    con enfoque en atención dispersa geoespacial.
    """
    
    def __init__(self, 
                 data_dir: str = "downloads/precipitation",
                 processed_dir: str = "processed_data",
                 sequence_length: int = 48,  # 24 horas con datos cada 30 min
                 prediction_horizon: int = 1,  # Predecir el siguiente paso
                 spatial_resolution: int = 50):  # Reducir resolución espacial
        """
        Inicializa el preprocesador.
        
        Args:
            data_dir: Directorio con archivos HDF5 descargados
            processed_dir: Directorio para datos procesados
            sequence_length: Longitud de secuencia temporal de entrada
            prediction_horizon: Pasos a predecir hacia el futuro
            spatial_resolution: Resolución espacial reducida (cada N píxeles)
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.spatial_resolution = spatial_resolution
        
        # Crear directorios si no existen
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(f"{processed_dir}/sequences", exist_ok=True)
        os.makedirs(f"{processed_dir}/metadata", exist_ok=True)
        
        # Metadatos para el procesamiento
        self.scaler = StandardScaler()
        self.metadata = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'spatial_resolution': spatial_resolution,
            'variable_names': [],
            'spatial_shape': None,
            'temporal_range': None
        }
    
    def read_hdf5_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Lee un archivo HDF5 y extrae las variables principales.
        
        Args:
            filepath: Ruta al archivo HDF5
            
        Returns:
            Diccionario con variables extraídas
        """
        variables = {}
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Función recursiva para explorar la estructura HDF5
                def explore_group(group, path=""):
                    for key in group.keys():
                        item = group[key]
                        current_path = f"{path}/{key}" if path else key
                        
                        if isinstance(item, h5py.Dataset):
                            # Es un dataset, extraerlo
                            data = item[:]
                            if data.ndim >= 2:  # Solo datos 2D o superiores
                                variables[current_path] = data
                        elif isinstance(item, h5py.Group):
                            # Es un grupo, explorar recursivamente
                            explore_group(item, current_path)
                
                explore_group(f)
                
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")
            
        return variables
    
    def extract_spatial_frame(self, data: np.ndarray) -> np.ndarray:
        """
        Extrae y procesa un "fotograma" espacial de los datos.
        Aplica reducción de resolución y limpieza de datos.
        
        Args:
            data: Array 2D o 3D con datos geoespaciales
            
        Returns:
            Array 2D procesado
        """
        # Si es 3D, tomar la primera dimensión temporal o banda
        if data.ndim == 3:
            if data.shape[0] < data.shape[2]:
                data = data[0, :, :]  # Primera banda/tiempo
            else:
                data = data[:, :, 0]  # Primera banda en última dimensión
        
        # Reducir resolución espacial para eficiencia computacional
        if self.spatial_resolution > 1:
            data = data[::self.spatial_resolution, ::self.spatial_resolution]
        
        # Limpiar valores no válidos
        data = np.where(np.isfinite(data), data, 0)
        data = np.where(data > -9999, data, 0)  # Reemplazar valores faltantes típicos
        
        return data
    
    def create_temporal_sequence(self, file_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Crea secuencias temporales a partir de múltiples archivos HDF5.
        
        Args:
            file_paths: Lista de rutas a archivos HDF5 ordenados temporalmente
            
        Returns:
            (secuencias_temporales, timestamps)
        """
        frames = []
        timestamps = []
        variable_names = set()
        
        print(f"Procesando {len(file_paths)} archivos...")
        
        for i, filepath in enumerate(file_paths):
            if i % 10 == 0:
                print(f"Procesando archivo {i+1}/{len(file_paths)}")
            
            variables = self.read_hdf5_file(filepath)
            
            if not variables:
                continue
            
            # Extraer timestamp del nombre del archivo
            filename = os.path.basename(filepath)
            # Formato típico: 3B-HHR.MS.MRG.3IMERG.20230101-S000000-E002959.0000.V07B.HDF5
            try:
                date_part = filename.split('.')[4]  # 20230101-S000000-E002959
                date_str = date_part.split('-')[0]  # 20230101
                time_str = date_part.split('-')[1][1:]  # 000000
                
                timestamp = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                timestamps.append(timestamp)
            except:
                timestamps.append(datetime.now() + timedelta(minutes=i*30))
            
            # Combinar todas las variables en un solo frame
            combined_frame = None
            target_shape = None
            
            for var_name, data in variables.items():
                variable_names.add(var_name)
                processed_data = self.extract_spatial_frame(data)
                
                # Establecer forma objetivo basada en el primer dato válido
                if target_shape is None and processed_data.size > 1:
                    target_shape = processed_data.shape
                
                # Redimensionar datos para que coincidan con la forma objetivo
                if target_shape is not None and processed_data.shape != target_shape:
                    # Si es un escalar o vector, expandir
                    if processed_data.size == 1:
                        processed_data = np.full(target_shape, processed_data.item())
                    else:
                        # Redimensionar manteniendo aspecto si es posible
                        try:
                            processed_data = np.resize(processed_data, target_shape)
                        except:
                            # Fallback: crear array con la forma correcta
                            processed_data = np.full(target_shape, np.mean(processed_data))
                
                if combined_frame is None:
                    combined_frame = processed_data
                else:
                    # Ahora todas las formas deberían coincidir
                    if processed_data.shape == combined_frame.shape:
                        if len(combined_frame.shape) == 2:
                            combined_frame = np.stack([combined_frame, processed_data], axis=0)
                        else:
                            combined_frame = np.concatenate([combined_frame[np.newaxis], 
                                                           processed_data[np.newaxis]], axis=0)
                    else:
                        print(f"Advertencia: saltando variable {var_name} por forma incompatible")
            
            if combined_frame is not None:
                # Asegurar que el frame tiene al menos 2 dimensiones
                if combined_frame.ndim == 1:
                    combined_frame = combined_frame.reshape(1, -1)
                elif combined_frame.ndim > 3:
                    # Si hay demasiadas dimensiones, aplanar a 2D
                    combined_frame = combined_frame.reshape(combined_frame.shape[0], -1)
                
                frames.append(combined_frame)
        
        if not frames:
            raise ValueError("No se pudieron procesar archivos válidos")
        
        # Convertir a array numpy
        frames_array = np.array(frames)
        self.metadata['variable_names'] = list(variable_names)
        self.metadata['spatial_shape'] = frames_array.shape[1:]
        self.metadata['temporal_range'] = (min(timestamps), max(timestamps))
        
        print(f"Secuencia creada: {frames_array.shape}")
        print(f"Variables encontradas: {variable_names}")
        
        return frames_array, timestamps
    
    def flatten_spatial_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Convierte fotogramas 2D/3D en vectores 1D para el modelo Transformer.
        
        Args:
            frames: Array de fotogramas (tiempo, [canales], altura, anchura)
            
        Returns:
            Array aplanado (tiempo, vector_espacial)
        """
        original_shape = frames.shape
        
        if len(original_shape) == 3:  # (tiempo, altura, anchura)
            flattened = frames.reshape(frames.shape[0], -1)
        elif len(original_shape) == 4:  # (tiempo, canales, altura, anchura)
            flattened = frames.reshape(frames.shape[0], -1)
        else:
            raise ValueError(f"Formato de frames no soportado: {original_shape}")
        
        print(f"Frames aplanados: {original_shape} -> {flattened.shape}")
        return flattened
    
    def create_training_sequences(self, flattened_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias de entrenamiento siguiendo el patrón:
        Entrada: [vector_tiempo_1, ..., vector_tiempo_N]
        Salida: vector_tiempo_(N+prediction_horizon)
        
        Args:
            flattened_data: Datos aplanados (tiempo, vector_espacial)
            
        Returns:
            (X, y) - Secuencias de entrada y etiquetas de salida
        """
        X, y = [], []
        
        for i in range(len(flattened_data) - self.sequence_length - self.prediction_horizon + 1):
            # Secuencia de entrada
            input_sequence = flattened_data[i:i + self.sequence_length]
            
            # Etiqueta de salida (predicción futura)
            target = flattened_data[i + self.sequence_length + self.prediction_horizon - 1]
            
            X.append(input_sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Secuencias de entrenamiento creadas: X={X.shape}, y={y.shape}")
        return X, y
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normaliza los datos usando StandardScaler.
        
        Args:
            X: Secuencias de entrada
            y: Etiquetas de salida
            
        Returns:
            (X_normalized, y_normalized)
        """
        # Aplanar X para normalización
        original_shape = X.shape
        X_flattened = X.reshape(-1, X.shape[-1])
        
        # Ajustar el scaler con todos los datos
        all_data = np.vstack([X_flattened, y])
        self.scaler.fit(all_data)
        
        # Normalizar
        X_normalized = self.scaler.transform(X_flattened).reshape(original_shape)
        y_normalized = self.scaler.transform(y)
        
        print(f"Datos normalizados: X std={X_normalized.std():.3f}, y std={y_normalized.std():.3f}")
        return X_normalized, y_normalized
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.
        
        Args:
            X: Secuencias de entrada normalizadas
            y: Etiquetas de salida normalizadas
            test_size: Proporción para conjunto de prueba
            val_size: Proporción para conjunto de validación (del resto)
            
        Returns:
            Diccionario con splits de datos
        """
        # División temporal para evitar data leakage
        n_samples = len(X)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - val_size))
        
        splits = {
            'X_train': X[:val_start],
            'y_train': y[:val_start],
            'X_val': X[val_start:test_start],
            'y_val': y[val_start:test_start],
            'X_test': X[test_start:],
            'y_test': y[test_start:]
        }
        
        print("División de datos:")
        for key, value in splits.items():
            print(f"  {key}: {value.shape}")
        
        return splits
    
    def save_processed_data(self, data_splits: Dict[str, np.ndarray], 
                          timestamps: List[str]) -> None:
        """
        Guarda los datos procesados y metadatos.
        
        Args:
            data_splits: Diccionario con splits de datos
            timestamps: Lista de timestamps
        """
        # Guardar splits de datos
        for name, data in data_splits.items():
            np.save(f"{self.processed_dir}/sequences/{name}.npy", data)
        
        # Guardar scaler
        with open(f"{self.processed_dir}/metadata/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Guardar metadatos
        self.metadata['timestamps'] = [t.isoformat() if hasattr(t, 'isoformat') else str(t) 
                                     for t in timestamps]
        
        with open(f"{self.processed_dir}/metadata/metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Datos guardados en {self.processed_dir}")
    
    def process_all_data(self) -> Dict[str, np.ndarray]:
        """
        Ejecuta todo el pipeline de preprocesamiento.
        
        Returns:
            Diccionario con splits de datos procesados
        """
        print("=== INICIANDO PREPROCESAMIENTO DE DATOS ===")
        
        # 1. Obtener lista de archivos HDF5
        file_pattern = os.path.join(self.data_dir, "*.HDF5")
        hdf5_files = sorted(glob.glob(file_pattern))
        
        if not hdf5_files:
            raise ValueError(f"No se encontraron archivos HDF5 en {self.data_dir}")
        
        print(f"Encontrados {len(hdf5_files)} archivos HDF5")
        
        # 2. Crear secuencias temporales
        frames, timestamps = self.create_temporal_sequence(hdf5_files)
        
        # 3. Aplanar fotogramas espaciales
        flattened_data = self.flatten_spatial_frames(frames)
        
        # 4. Crear secuencias de entrenamiento
        X, y = self.create_training_sequences(flattened_data)
        
        # 5. Normalizar datos
        X_norm, y_norm = self.normalize_data(X, y)
        
        # 6. Dividir en train/val/test
        data_splits = self.split_data(X_norm, y_norm)
        
        # 7. Guardar datos procesados
        self.save_processed_data(data_splits, timestamps)
        
        print("=== PREPROCESAMIENTO COMPLETADO ===")
        return data_splits


def main():
    """Función principal para ejecutar el preprocesamiento."""
    
    # Configuración
    preprocessor = EarthDataPreprocessor(
        data_dir="downloads/precipitation",
        processed_dir="processed_data",
        sequence_length=5,  # Reducido para trabajar con 10 archivos
        prediction_horizon=1,  # Predecir próximos 30 min
        spatial_resolution=20  # Reducir resolución para eficiencia
    )
    
    try:
        # Ejecutar preprocesamiento completo
        data_splits = preprocessor.process_all_data()
        
        # Mostrar estadísticas finales
        print("\n=== ESTADÍSTICAS FINALES ===")
        print(f"Forma de secuencias de entrada: {data_splits['X_train'].shape}")
        print(f"Forma de etiquetas: {data_splits['y_train'].shape}")
        print(f"Total de muestras de entrenamiento: {len(data_splits['X_train'])}")
        print(f"Total de muestras de validación: {len(data_splits['X_val'])}")
        print(f"Total de muestras de prueba: {len(data_splits['X_test'])}")
        
        # Verificar que no hay valores NaN
        for name, data in data_splits.items():
            nan_count = np.isnan(data).sum()
            print(f"{name}: {nan_count} valores NaN")
        
        print("\n¡Datos listos para entrenamiento del modelo!")
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        raise


if __name__ == "__main__":
    main()