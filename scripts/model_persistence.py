#!/usr/bin/env python3
"""
Sistema de Persistencia de Modelos con Soporte Múltiple
=====================================================

Implementa guardado/carga de modelos en múltiples formatos:
- Pickle nativo (rápido)
- JSON/NumPy (interoperabilidad)
- HDF5 (datasets grandes)
- SafeTensors (compatibilidad PyTorch/HuggingFace)

Incluye:
- Metadatos del modelo
- Versionado
- Validación de integridad
- Compresión automática
"""

import os
import json
import pickle
import hashlib
import gzip
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py no disponible. Funcionalidad HDF5 deshabilitada.")

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn("safetensors no disponible. Instalalo con: pip install safetensors")


class ModelCheckpoint:
    """Información de checkpoint del modelo."""
    
    def __init__(self, 
                 model_type: str,
                 version: str = "1.0.0",
                 metadata: Optional[Dict] = None):
        self.model_type = model_type
        self.version = version
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}
        self.checksum = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'model_type': self.model_type,
            'version': self.version,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCheckpoint':
        """Crea desde diccionario."""
        checkpoint = cls(
            model_type=data['model_type'],
            version=data.get('version', '1.0.0'),
            metadata=data.get('metadata', {})
        )
        checkpoint.timestamp = data.get('timestamp', checkpoint.timestamp)
        checkpoint.checksum = data.get('checksum')
        return checkpoint


class ModelPersistence:
    """Sistema de persistencia de modelos con múltiples formatos."""
    
    SUPPORTED_FORMATS = ['pickle', 'json_numpy', 'hdf5', 'safetensors']
    
    def __init__(self, compression: bool = True, validate_integrity: bool = True):
        """
        Inicializar sistema de persistencia.
        
        Args:
            compression: Si comprimir archivos automáticamente
            validate_integrity: Si validar integridad con checksums
        """
        self.compression = compression
        self.validate_integrity = validate_integrity
    
    def _compute_checksum(self, data: bytes) -> str:
        """Calcula checksum SHA256 de los datos."""
        return hashlib.sha256(data).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Comprime datos si está habilitado."""
        if self.compression:
            return gzip.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprime datos si es necesario."""
        try:
            return gzip.decompress(data)
        except gzip.BadGzipFile:
            return data  # No estaba comprimido
    
    def save_model_pickle(self, 
                         model: Any, 
                         filepath: str,
                         metadata: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Guarda modelo en formato Pickle.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta del archivo
            metadata: Metadatos adicionales
            
        Returns:
            Checkpoint con información del guardado
        """
        # Preparar metadatos
        checkpoint = ModelCheckpoint(
            model_type=type(model).__name__,
            metadata=metadata or {}
        )
        
        # Serializar modelo
        model_data = pickle.dumps(model)
        
        # Calcular checksum
        if self.validate_integrity:
            checkpoint.checksum = self._compute_checksum(model_data)
        
        # Comprimir si está habilitado
        compressed_data = self._compress_data(model_data)
        
        # Crear paquete completo
        package = {
            'checkpoint': checkpoint.to_dict(),
            'model_data': compressed_data
        }
        
        # Guardar archivo
        with open(filepath, 'wb') as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Modelo guardado en formato Pickle: {filepath}")
        print(f"Tamaño: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return checkpoint
    
    def load_model_pickle(self, filepath: str) -> Tuple[Any, ModelCheckpoint]:
        """
        Carga modelo desde formato Pickle.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Tupla (modelo, checkpoint)
        """
        with open(filepath, 'rb') as f:
            package = pickle.load(f)
        
        # Extraer checkpoint
        checkpoint = ModelCheckpoint.from_dict(package['checkpoint'])
        
        # Descomprimir datos del modelo
        model_data = self._decompress_data(package['model_data'])
        
        # Validar integridad
        if self.validate_integrity and checkpoint.checksum:
            computed_checksum = self._compute_checksum(model_data)
            if computed_checksum != checkpoint.checksum:
                raise ValueError("Checksum no coincide. Archivo posiblemente corrupto.")
        
        # Deserializar modelo
        model = pickle.loads(model_data)
        
        print(f"Modelo cargado desde Pickle: {filepath}")
        print(f"Tipo: {checkpoint.model_type}, Versión: {checkpoint.version}")
        
        return model, checkpoint
    
    def save_model_numpy(self, 
                        model: Any, 
                        filepath: str,
                        metadata: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Guarda modelo en formato JSON + NumPy.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta base (sin extensión)
            metadata: Metadatos adicionales
            
        Returns:
            Checkpoint con información del guardado
        """
        # Extraer atributos numpy del modelo
        numpy_arrays = {}
        model_config = {}
        
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr_value = getattr(model, attr_name)
                
                if isinstance(attr_value, np.ndarray):
                    numpy_arrays[attr_name] = attr_value
                elif isinstance(attr_value, (int, float, str, bool, list, dict)):
                    model_config[attr_name] = attr_value
        
        # Preparar checkpoint
        checkpoint = ModelCheckpoint(
            model_type=type(model).__name__,
            metadata=metadata or {}
        )
        
        # Guardar configuración JSON
        config_file = f"{filepath}_config.json"
        full_config = {
            'checkpoint': checkpoint.to_dict(),
            'model_config': model_config,
            'numpy_arrays': list(numpy_arrays.keys())
        }
        
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        # Guardar arrays NumPy
        arrays_file = f"{filepath}_arrays.npz"
        if self.compression:
            np.savez_compressed(arrays_file, **numpy_arrays)
        else:
            np.savez(arrays_file, **numpy_arrays)
        
        print(f"Modelo guardado en formato NumPy:")
        print(f"  Configuración: {config_file}")
        print(f"  Arrays: {arrays_file}")
        
        return checkpoint
    
    def save_model_hdf5(self, 
                       model: Any, 
                       filepath: str,
                       metadata: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Guarda modelo en formato HDF5.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta del archivo
            metadata: Metadatos adicionales
            
        Returns:
            Checkpoint con información del guardado
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py no disponible. Instalar con: pip install h5py")
        
        checkpoint = ModelCheckpoint(
            model_type=type(model).__name__,
            metadata=metadata or {}
        )
        
        with h5py.File(filepath, 'w') as f:
            # Guardar metadatos
            meta_group = f.create_group('metadata')
            for key, value in checkpoint.to_dict().items():
                if isinstance(value, dict):
                    sub_group = meta_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.attrs[sub_key] = str(sub_value)
                else:
                    meta_group.attrs[key] = str(value)
            
            # Guardar datos del modelo
            model_group = f.create_group('model')
            
            # Serializar modelo completo
            model_bytes = pickle.dumps(model)
            compressed_bytes = self._compress_data(model_bytes)
            
            model_group.create_dataset('serialized_model', 
                                     data=np.frombuffer(compressed_bytes, dtype=np.uint8))
            
            # Guardar arrays NumPy por separado para acceso directo
            arrays_group = f.create_group('arrays')
            for attr_name in dir(model):
                if not attr_name.startswith('_'):
                    attr_value = getattr(model, attr_name)
                    if isinstance(attr_value, np.ndarray):
                        arrays_group.create_dataset(attr_name, data=attr_value)
        
        print(f"Modelo guardado en formato HDF5: {filepath}")
        print(f"Tamaño: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return checkpoint
    
    def save_model_safetensors(self, 
                              model: Any, 
                              filepath: str,
                              metadata: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Guarda modelo en formato SafeTensors.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta del archivo
            metadata: Metadatos adicionales
            
        Returns:
            Checkpoint con información del guardado
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors no disponible. Instalar con: pip install safetensors")
        
        # Extraer arrays NumPy
        tensors = {}
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr_value = getattr(model, attr_name)
                if isinstance(attr_value, np.ndarray):
                    # SafeTensors requiere nombres únicos y tipos específicos
                    tensors[f"model_{attr_name}"] = attr_value.astype(np.float32)
        
        # Preparar metadatos
        checkpoint = ModelCheckpoint(
            model_type=type(model).__name__,
            metadata=metadata or {}
        )
        
        # Serializar configuración no-tensor
        model_config = {}
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr_value = getattr(model, attr_name)
                if not isinstance(attr_value, np.ndarray):
                    if isinstance(attr_value, (int, float, str, bool)):
                        model_config[attr_name] = attr_value
        
        # Combinar metadatos
        all_metadata = {
            **checkpoint.to_dict(),
            'model_config': model_config
        }
        
        # Convertir metadatos a strings (SafeTensors requirement)
        string_metadata = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in all_metadata.items()}
        
        # Guardar con SafeTensors
        safetensors_save(tensors, filepath, metadata=string_metadata)
        
        print(f"Modelo guardado en formato SafeTensors: {filepath}")
        print(f"Tamaño: {os.path.getsize(filepath) / 1024:.1f} KB")
        print(f"Arrays guardados: {list(tensors.keys())}")
        
        return checkpoint
    
    def save_model(self, 
                   model: Any, 
                   filepath: str, 
                   format: str = 'pickle',
                   metadata: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Guarda modelo en el formato especificado.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta del archivo
            format: Formato ('pickle', 'json_numpy', 'hdf5', 'safetensors')
            metadata: Metadatos adicionales
            
        Returns:
            Checkpoint con información del guardado
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Formato no soportado: {format}. "
                           f"Soportados: {self.SUPPORTED_FORMATS}")
        
        # Añadir metadatos automáticos
        auto_metadata = {
            'model_class': type(model).__name__,
            'save_format': format,
            'compressed': self.compression,
            'file_size_hint': 'unknown'
        }
        
        if metadata:
            auto_metadata.update(metadata)
        
        if format == 'pickle':
            return self.save_model_pickle(model, filepath, auto_metadata)
        elif format == 'json_numpy':
            return self.save_model_numpy(model, filepath, auto_metadata)
        elif format == 'hdf5':
            return self.save_model_hdf5(filepath, auto_metadata)
        elif format == 'safetensors':
            return self.save_model_safetensors(model, filepath, auto_metadata)
    
    def load_model(self, 
                   filepath: str, 
                   format: Optional[str] = None) -> Tuple[Any, ModelCheckpoint]:
        """
        Carga modelo detectando formato automáticamente.
        
        Args:
            filepath: Ruta del archivo
            format: Formato específico (opcional, auto-detecta si None)
            
        Returns:
            Tupla (modelo, checkpoint)
        """
        if format is None:
            # Auto-detectar formato
            if filepath.endswith('.pkl'):
                format = 'pickle'
            elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
                format = 'hdf5'
            elif filepath.endswith('.safetensors'):
                format = 'safetensors'
            elif '_config.json' in filepath or os.path.exists(f"{filepath}_config.json"):
                format = 'json_numpy'
            else:
                # Intentar pickle por defecto
                format = 'pickle'
        
        if format == 'pickle':
            return self.load_model_pickle(filepath)
        else:
            raise NotImplementedError(f"Carga para formato {format} aún no implementada")


def demonstrate_model_persistence():
    """Demuestra el uso del sistema de persistencia."""
    print("=" * 50)
    print("DEMOSTRACIÓN DE PERSISTENCIA DE MODELOS")
    print("=" * 50)
    
    # Simular modelo simple
    class SimpleModel:
        def __init__(self):
            self.weights = np.random.randn(10, 5)
            self.bias = np.random.randn(5)
            self.config = {'learning_rate': 0.01, 'epochs': 100}
            self.is_fitted = True
        
        def predict(self, X):
            return X @ self.weights + self.bias
    
    # Crear modelo
    model = SimpleModel()
    
    # Crear sistema de persistencia
    persistence = ModelPersistence(compression=True, validate_integrity=True)
    
    # Metadatos de ejemplo
    metadata = {
        'author': 'NASA Hackathon Team',
        'dataset': 'IMERG Precipitation',
        'performance': {'r2': 0.85, 'rmse': 0.012}
    }
    
    # Probar diferentes formatos
    for format_name in ['pickle', 'safetensors']:
        if format_name == 'safetensors' and not SAFETENSORS_AVAILABLE:
            print(f"Saltando {format_name} (no disponible)")
            continue
        
        print(f"\n--- Probando formato {format_name} ---")
        
        # Guardar
        filepath = f"demo_model.{format_name}"
        try:
            checkpoint = persistence.save_model(model, filepath, format_name, metadata)
            print(f"✓ Guardado exitoso")
            
            # Cargar
            loaded_model, loaded_checkpoint = persistence.load_model(filepath, format_name)
            print(f"✓ Carga exitosa")
            
            # Verificar que funciona
            test_input = np.random.randn(3, 10)
            original_pred = model.predict(test_input)
            loaded_pred = loaded_model.predict(test_input)
            
            if np.allclose(original_pred, loaded_pred):
                print("✓ Predicciones idénticas")
            else:
                print("✗ Predicciones diferentes")
            
            # Limpiar
            if os.path.exists(filepath):
                os.remove(filepath)
        
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == '__main__':
    demonstrate_model_persistence()