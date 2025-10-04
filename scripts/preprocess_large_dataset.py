#!/usr/bin/env python3
"""
Preprocesamiento de Dataset Grande
=================================

Versión optimizada para procesar grandes volúmenes de datos NASA IMERG
con procesamiento por lotes, manejo de memoria eficiente y paralelización.
"""

import os
import sys
import numpy as np
import h5py
import xarray as xr
from pathlib import Path
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
from datetime import datetime
import json

class LargeDatasetPreprocessor:
    """Preprocesador optimizado para datasets grandes."""
    
    def __init__(self, 
                 raw_data_dir="raw_data_large",
                 processed_dir="processed_data_large",
                 batch_size=50,
                 sequence_length=24,
                 prediction_horizon=6,
                 spatial_resolution=4,
                 max_workers=None):
        """
        Inicializar preprocesador para datasets grandes.
        
        Args:
            raw_data_dir: Directorio con archivos HDF5 crudos
            processed_dir: Directorio para datos procesados
            batch_size: Archivos a procesar por lote
            sequence_length: Longitud de secuencias temporales
            prediction_horizon: Horizonte de predicción
            spatial_resolution: Factor de reducción espacial
            max_workers: Procesos paralelos (None = auto)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.spatial_resolution = spatial_resolution
        
        # Configurar paralelización
        if max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)  # Máximo 8 procesos
        else:
            self.max_workers = max_workers
        
        # Crear directorios
        self.processed_dir.mkdir(exist_ok=True)
        (self.processed_dir / "batches").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
        
        print(f"🔧 Configuración del preprocesador:")
        print(f"   - Lote de procesamiento: {batch_size} archivos")
        print(f"   - Secuencia temporal: {sequence_length} frames")
        print(f"   - Horizonte predicción: {prediction_horizon} frames")
        print(f"   - Reducción espacial: {spatial_resolution}x")
        print(f"   - Procesos paralelos: {self.max_workers}")
    
    def get_hdf5_files(self):
        """Obtiene lista de archivos HDF5 ordenados por fecha."""
        hdf5_files = list(self.raw_data_dir.glob("*.HDF5"))
        
        if not hdf5_files:
            print(f"❌ No se encontraron archivos HDF5 en {self.raw_data_dir}")
            return []
        
        # Ordenar por fecha en el nombre del archivo
        hdf5_files.sort(key=lambda x: x.name)
        
        print(f"📂 Encontrados {len(hdf5_files)} archivos HDF5")
        return hdf5_files
    
    def estimate_memory_usage(self, sample_files):
        """Estima uso de memoria para el procesamiento."""
        if not sample_files:
            return 0, (0, 0)
        
        # Analizar archivo de muestra
        sample_file = sample_files[0]
        try:
            with h5py.File(sample_file, 'r') as f:
                # Buscar variable principal de precipitación
                precip_var = None
                for var_path in ['Grid/precipitation', 'Grid/precipitationCal']:
                    if var_path in f:
                        precip_var = f[var_path]
                        break
                
                if precip_var is not None:
                    shape = precip_var.shape
                    dtype = precip_var.dtype
                    
                    # Calcular tamaño original
                    original_size = np.prod(shape) * np.dtype(dtype).itemsize
                    
                    # Estimar después de reducción espacial
                    if len(shape) >= 2:
                        reduced_height = shape[-2] // self.spatial_resolution
                        reduced_width = shape[-1] // self.spatial_resolution
                        reduced_shape = shape[:-2] + (reduced_height, reduced_width)
                        reduced_size = np.prod(reduced_shape) * np.dtype(dtype).itemsize
                    else:
                        reduced_size = original_size
                    
                    spatial_shape = (reduced_height, reduced_width) if len(shape) >= 2 else shape
                    
                    print(f"📊 Análisis de memoria:")
                    print(f"   - Forma espacial estimada: {spatial_shape}")
                    print(f"   - Tamaño por archivo: {original_size / (1024**2):.1f} MB")
                    print(f"   - Después de reducción: {reduced_size / (1024**2):.1f} MB")
                    
                    return reduced_size, spatial_shape
                
        except Exception as e:
            print(f"⚠️ Error analizando archivo muestra: {e}")
        
        return 0, (0, 0)
    
    def process_single_file(self, file_path):
        """
        Procesa un archivo HDF5 individual.
        
        Args:
            file_path: Ruta al archivo HDF5
            
        Returns:
            Array numpy procesado o None si hay error
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Buscar variable de precipitación
                data_array = None
                
                for var_path in ['Grid/precipitation', 'Grid/precipitationCal', 
                               'Grid/Intermediate/MWprecipitation']:
                    if var_path in f:
                        data_array = f[var_path][:]
                        break
                
                if data_array is None:
                    print(f"⚠️ No se encontró variable de precipitación en {file_path.name}")
                    return None
                
                # Procesar dimensiones
                if data_array.ndim == 2:
                    # Ya es 2D (lat, lon)
                    processed = data_array
                elif data_array.ndim == 3:
                    # Tomar primer frame temporal o hacer promedio
                    processed = data_array[0] if data_array.shape[0] == 1 else np.mean(data_array, axis=0)
                else:
                    print(f"⚠️ Forma de datos no soportada: {data_array.shape}")
                    return None
                
                # Reducir resolución espacial
                if self.spatial_resolution > 1:
                    h, w = processed.shape
                    new_h = h // self.spatial_resolution
                    new_w = w // self.spatial_resolution
                    
                    # Reshape y promediar
                    processed = processed[:new_h*self.spatial_resolution, :new_w*self.spatial_resolution]
                    processed = processed.reshape(new_h, self.spatial_resolution, 
                                                new_w, self.spatial_resolution).mean(axis=(1, 3))
                
                # Manejar valores faltantes
                processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
                
                return processed.astype(np.float32)
                
        except Exception as e:
            print(f"❌ Error procesando {file_path.name}: {e}")
            return None
    
    def process_batch_parallel(self, file_batch, batch_idx):
        """
        Procesa un lote de archivos en paralelo.
        
        Args:
            file_batch: Lista de archivos para procesar
            batch_idx: Índice del lote
            
        Returns:
            Array numpy con datos del lote
        """
        print(f"🔄 Procesando lote {batch_idx + 1} ({len(file_batch)} archivos)")
        
        # Usar ProcessPoolExecutor para paralelización
        batch_data = []
        successful = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar tareas
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in file_batch
            }
            
            # Recoger resultados manteniendo orden
            results = {}
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=300)  # 5 minutos timeout
                    if result is not None:
                        results[file_path] = result
                        successful += 1
                    else:
                        print(f"⚠️ Archivo {file_path.name} retornó None")
                except Exception as e:
                    print(f"❌ Error procesando {file_path.name}: {e}")
        
        # Reconstruir orden original
        for file_path in file_batch:
            if file_path in results:
                batch_data.append(results[file_path])
        
        if batch_data:
            batch_array = np.stack(batch_data, axis=0)
            print(f"✅ Lote {batch_idx + 1} completado: {successful}/{len(file_batch)} archivos, forma {batch_array.shape}")
            return batch_array
        else:
            print(f"❌ Lote {batch_idx + 1} falló completamente")
            return None
    
    def create_sequences_from_batch(self, batch_data, batch_idx):
        """
        Crea secuencias temporales desde datos de lote.
        
        Args:
            batch_data: Array numpy (time, height, width)
            batch_idx: Índice del lote
            
        Returns:
            Tupla (X, y) con secuencias de entrenamiento
        """
        if batch_data is None or len(batch_data) < self.sequence_length + self.prediction_horizon:
            print(f"⚠️ Lote {batch_idx + 1} insuficiente para crear secuencias")
            return None, None
        
        # Aplanar dimensiones espaciales
        n_time, height, width = batch_data.shape
        spatial_dim = height * width
        flattened_data = batch_data.reshape(n_time, spatial_dim)
        
        # Crear secuencias
        X_sequences = []
        y_sequences = []
        
        for i in range(n_time - self.sequence_length - self.prediction_horizon + 1):
            # Secuencia de entrada
            X_seq = flattened_data[i:i + self.sequence_length]
            
            # Target (después del horizonte de predicción)
            y_target = flattened_data[i + self.sequence_length + self.prediction_horizon - 1]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_target)
        
        if X_sequences:
            X = np.stack(X_sequences, axis=0)
            y = np.stack(y_sequences, axis=0)
            
            print(f"📋 Lote {batch_idx + 1}: creadas {len(X_sequences)} secuencias")
            print(f"   - X: {X.shape}, y: {y.shape}")
            
            return X, y
        else:
            return None, None
    
    def save_batch_results(self, X, y, batch_idx, spatial_shape):
        """Guarda resultados de lote procesado."""
        if X is None or y is None:
            return
        
        batch_file = self.processed_dir / "batches" / f"batch_{batch_idx:04d}.npz"
        
        np.savez_compressed(
            batch_file,
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            spatial_shape=np.array(spatial_shape),
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            batch_idx=batch_idx
        )
        
        print(f"💾 Guardado lote {batch_idx + 1}: {batch_file.name}")
    
    def process_large_dataset(self):
        """Procesa dataset grande por lotes."""
        print("🚀 INICIANDO PROCESAMIENTO DE DATASET GRANDE")
        print("=" * 60)
        
        # Obtener archivos
        hdf5_files = self.get_hdf5_files()
        if not hdf5_files:
            return
        
        # Estimar memoria y forma espacial
        file_size, spatial_shape = self.estimate_memory_usage(hdf5_files[:5])
        
        # Verificar memoria disponible
        available_memory = psutil.virtual_memory().available
        estimated_batch_memory = file_size * self.batch_size * 4  # Factor de seguridad
        
        if estimated_batch_memory > available_memory * 0.8:
            recommended_batch = max(1, int(available_memory * 0.6 / file_size))
            print(f"⚠️ Lote demasiado grande para memoria disponible")
            print(f"   Recomendado: {recommended_batch} archivos por lote")
            
            response = input("¿Continuar con lote recomendado? (y/N): ")
            if response.lower() == 'y':
                self.batch_size = recommended_batch
            else:
                print("❌ Procesamiento cancelado")
                return
        
        # Procesar por lotes
        total_batches = (len(hdf5_files) + self.batch_size - 1) // self.batch_size
        total_sequences = 0
        
        print(f"📊 Configuración final:")
        print(f"   - Total archivos: {len(hdf5_files)}")
        print(f"   - Lotes a procesar: {total_batches}")
        print(f"   - Archivos por lote: {self.batch_size}")
        
        start_time = datetime.now()
        
        for batch_idx in range(total_batches):
            print(f"\n--- Lote {batch_idx + 1}/{total_batches} ---")
            
            # Definir archivos del lote
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(hdf5_files))
            file_batch = hdf5_files[start_idx:end_idx]
            
            # Procesar lote
            batch_data = self.process_batch_parallel(file_batch, batch_idx)
            
            if batch_data is not None:
                # Crear secuencias
                X, y = self.create_sequences_from_batch(batch_data, batch_idx)
                
                if X is not None and y is not None:
                    # Guardar resultados
                    self.save_batch_results(X, y, batch_idx, spatial_shape)
                    total_sequences += len(X)
            
            # Limpiar memoria
            del batch_data
            gc.collect()
        
        # Consolidar metadatos
        self.save_processing_metadata(hdf5_files, spatial_shape, total_sequences, start_time)
        
        print("\n" + "=" * 60)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"📊 Estadísticas finales:")
        print(f"   - Archivos procesados: {len(hdf5_files)}")
        print(f"   - Lotes generados: {total_batches}")
        print(f"   - Secuencias totales: {total_sequences}")
        print(f"   - Forma espacial: {spatial_shape}")
        print(f"   - Tiempo total: {datetime.now() - start_time}")
        
        if total_sequences > 0:
            print(f"\n🎉 Dataset listo para entrenamiento!")
            print(f"📁 Datos en: {self.processed_dir}")
            print(f"\n🚀 PRÓXIMO PASO:")
            print(f"   python scripts/train_large_model.py")
    
    def save_processing_metadata(self, source_files, spatial_shape, total_sequences, start_time):
        """Guarda metadatos del procesamiento."""
        metadata = {
            'processing_config': {
                'batch_size': self.batch_size,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'spatial_resolution': self.spatial_resolution,
                'max_workers': self.max_workers
            },
            'dataset_info': {
                'total_source_files': len(source_files),
                'total_sequences': total_sequences,
                'spatial_shape': spatial_shape,
                'processed_at': start_time.isoformat(),
                'processing_time': str(datetime.now() - start_time)
            },
            'source_files': [str(f.name) for f in source_files]
        }
        
        metadata_file = self.processed_dir / "metadata" / "processing_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadatos guardados: {metadata_file}")


def main():
    """Función principal con argumentos."""
    parser = argparse.ArgumentParser(description='Procesar dataset grande NASA IMERG')
    
    parser.add_argument('--raw-dir', default='raw_data_large',
                       help='Directorio con archivos HDF5')
    parser.add_argument('--output-dir', default='processed_data_large',
                       help='Directorio para datos procesados')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Archivos por lote')
    parser.add_argument('--sequence-length', type=int, default=24,
                       help='Longitud secuencia temporal')
    parser.add_argument('--prediction-horizon', type=int, default=6,
                       help='Horizonte de predicción')
    parser.add_argument('--spatial-resolution', type=int, default=4,
                       help='Factor reducción espacial')
    parser.add_argument('--workers', type=int, default=None,
                       help='Procesos paralelos')
    
    args = parser.parse_args()
    
    # Crear preprocesador
    preprocessor = LargeDatasetPreprocessor(
        raw_data_dir=args.raw_dir,
        processed_dir=args.output_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        spatial_resolution=args.spatial_resolution,
        max_workers=args.workers
    )
    
    # Procesar dataset
    preprocessor.process_large_dataset()


if __name__ == "__main__":
    main()