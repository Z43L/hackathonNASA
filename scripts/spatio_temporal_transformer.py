#!/usr/bin/env python3
"""
Arquitectura Spatio-Temporal Transformer con Atención Dispersa para Datos Geoespaciales
======================================================================================

Implementación inspirada en DeepSeek's Sparse Attention (DSA) adaptada para datos
meteorológicos y geoespaciales. Esta implementación usa numpy y scikit-learn como
base, con una estructura modular que permite migrar fácilmente a PyTorch.

Componentes principales:
1. SpatialEmbedding: Embeddings posicionales para coordenadas geográficas
2. TemporalEmbedding: Embeddings temporales para secuencias
3. SparseAttentionIndexer: "Indexador relámpago" para seleccionar píxeles relevantes
4. SpatioTemporalTransformer: Modelo principal con atención dispersa
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict, Optional, Union
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SpatialEmbedding:
    """
    Genera embeddings posicionales para coordenadas espaciales (lat, lon).
    Equivalente a position encoding en Transformers pero para datos geoespaciales.
    """
    
    def __init__(self, embedding_dim: int = 64, max_lat: float = 90.0, max_lon: float = 180.0):
        """
        Args:
            embedding_dim: Dimensión del embedding espacial
            max_lat: Latitud máxima (grados)
            max_lon: Longitud máxima (grados)
        """
        self.embedding_dim = embedding_dim
        self.max_lat = max_lat
        self.max_lon = max_lon
        
        # Frecuencias para encoding sinusoidal
        self.freqs = np.logspace(0, np.log10(10000), embedding_dim // 4)
    
    def encode_position(self, lat: float, lon: float) -> np.ndarray:
        """
        Codifica una posición (lat, lon) en un embedding sinusoidal.
        
        Args:
            lat: Latitud normalizada [-1, 1]
            lon: Longitud normalizada [-1, 1]
            
        Returns:
            Vector de embedding espacial
        """
        # Normalizar coordenadas
        lat_norm = lat / self.max_lat
        lon_norm = lon / self.max_lon
        
        # Encoding sinusoidal
        lat_enc = np.concatenate([
            np.sin(lat_norm * self.freqs),
            np.cos(lat_norm * self.freqs)
        ])
        
        lon_enc = np.concatenate([
            np.sin(lon_norm * self.freqs),
            np.cos(lon_norm * self.freqs)
        ])
        
        return np.concatenate([lat_enc, lon_enc])
    
    def create_spatial_grid(self, height: int, width: int, 
                          lat_range: Tuple[float, float] = (-90, 90),
                          lon_range: Tuple[float, float] = (-180, 180)) -> np.ndarray:
        """
        Crea embeddings espaciales para una grilla completa.
        
        Args:
            height: Altura de la grilla
            width: Anchura de la grilla
            lat_range: Rango de latitudes
            lon_range: Rango de longitudes
            
        Returns:
            Array de embeddings (height, width, embedding_dim)
        """
        lats = np.linspace(lat_range[0], lat_range[1], height)
        lons = np.linspace(lon_range[0], lon_range[1], width)
        
        embeddings = np.zeros((height, width, self.embedding_dim))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                embeddings[i, j] = self.encode_position(lat, lon)
        
        return embeddings


class TemporalEmbedding:
    """
    Genera embeddings temporales para secuencias de tiempo.
    Incluye información cíclica (hora del día, día del año) y posicional.
    """
    
    def __init__(self, embedding_dim: int = 64, max_sequence_length: int = 1000):
        """
        Args:
            embedding_dim: Dimensión del embedding temporal
            max_sequence_length: Longitud máxima de secuencia
        """
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        
        # Frecuencias para encoding posicional
        self.freqs = np.logspace(0, np.log10(10000), embedding_dim // 2)
    
    def encode_position(self, position: int) -> np.ndarray:
        """
        Codifica posición en secuencia temporal.
        
        Args:
            position: Posición en la secuencia
            
        Returns:
            Vector de embedding temporal posicional
        """
        pos_norm = position / self.max_sequence_length
        
        return np.concatenate([
            np.sin(pos_norm * self.freqs),
            np.cos(pos_norm * self.freqs)
        ])
    
    def encode_cyclical(self, hour: int = 0, day_of_year: int = 1) -> np.ndarray:
        """
        Codifica información temporal cíclica.
        
        Args:
            hour: Hora del día (0-23)
            day_of_year: Día del año (1-365)
            
        Returns:
            Vector de embedding temporal cíclico
        """
        # Encoding cíclico para hora
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Encoding cíclico para día del año
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        return np.array([hour_sin, hour_cos, day_sin, day_cos])
    
    def create_sequence_embeddings(self, sequence_length: int) -> np.ndarray:
        """
        Crea embeddings para toda una secuencia temporal.
        
        Args:
            sequence_length: Longitud de la secuencia
            
        Returns:
            Array de embeddings (sequence_length, embedding_dim)
        """
        embeddings = np.zeros((sequence_length, self.embedding_dim))
        
        for i in range(sequence_length):
            pos_emb = self.encode_position(i)
            # Asumir datos cada 30 minutos
            hour = (i * 0.5) % 24
            day = ((i * 0.5) // 24) % 365 + 1
            cyc_emb = self.encode_cyclical(int(hour), int(day))
            
            # Combinar embeddings posicionales y cíclicos
            if len(pos_emb) + len(cyc_emb) <= self.embedding_dim:
                combined = np.concatenate([pos_emb, cyc_emb])
                # Padding si es necesario
                if len(combined) < self.embedding_dim:
                    combined = np.pad(combined, (0, self.embedding_dim - len(combined)))
                embeddings[i] = combined[:self.embedding_dim]
            else:
                embeddings[i] = pos_emb[:self.embedding_dim]
        
        return embeddings


class SparseAttentionIndexer:
    """
    "Indexador Relámpago" para identificar píxeles/regiones relevantes.
    Implementa el concepto clave de atención dispersa de DeepSeek para datos geoespaciales.
    """
    
    def __init__(self, 
                 spatial_dim: int,
                 top_k_ratio: float = 0.1,
                 attention_heads: int = 8,
                 random_state: int = 42):
        """
        Args:
            spatial_dim: Dimensión del espacio aplanado
            top_k_ratio: Proporción de píxeles más relevantes a seleccionar
            attention_heads: Número de cabezas de atención
            random_state: Semilla para reproducibilidad
        """
        self.spatial_dim = spatial_dim
        self.top_k = int(spatial_dim * top_k_ratio)
        self.attention_heads = attention_heads
        self.random_state = random_state
        
        # Inicializar pesos de manera aleatoria (normalmente se aprenderían)
        np.random.seed(random_state)
        
        # Pesos para calcular relevancia de píxeles
        self.query_weights = np.random.randn(spatial_dim, attention_heads) * 0.1
        self.key_weights = np.random.randn(spatial_dim, attention_heads) * 0.1
        
        # Pesos para indexador rápido
        self.indexer_weights = np.random.randn(spatial_dim, 64) * 0.1
        self.indexer_bias = np.random.randn(64) * 0.1
        
        # Pesos de salida del indexador
        self.output_weights = np.random.randn(64, spatial_dim) * 0.1
    
    def compute_pixel_relevance(self, spatial_frame: np.ndarray, 
                              target_location: Optional[int] = None) -> np.ndarray:
        """
        Calcula la relevancia de cada píxel para la predicción.
        
        Args:
            spatial_frame: Frame espacial aplanado (spatial_dim,)
            target_location: Ubicación objetivo para predicción (opcional)
            
        Returns:
            Scores de relevancia para cada píxel
        """
        # Query: representa qué estamos buscando
        if target_location is not None:
            query = np.zeros_like(spatial_frame)
            query[target_location] = 1.0
        else:
            # Query promedio sobre toda la región
            query = np.ones_like(spatial_frame) / len(spatial_frame)
        
        # Keys: representan el contenido de cada píxel
        keys = spatial_frame
        
        # Calcular attention scores
        relevance_scores = np.zeros(self.spatial_dim)
        
        for head in range(self.attention_heads):
            # Proyecciones lineales simuladas
            q_proj = np.dot(query, self.query_weights[:, head])
            k_proj = keys * self.key_weights[:, head]
            
            # Attention scores simplificados
            scores = k_proj * q_proj
            relevance_scores += scores
        
        # Normalizar scores
        relevance_scores = relevance_scores / self.attention_heads
        
        return relevance_scores
    
    def select_top_k_pixels(self, relevance_scores: np.ndarray) -> np.ndarray:
        """
        Selecciona los k píxeles más relevantes.
        
        Args:
            relevance_scores: Scores de relevancia
            
        Returns:
            Índices de píxeles seleccionados
        """
        # Obtener índices de los top-k elementos
        top_k_indices = np.argsort(relevance_scores)[-self.top_k:]
        
        return top_k_indices
    
    def lightning_indexer(self, spatial_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementa el "indexador relámpago" - versión rápida de selección.
        
        Args:
            spatial_frame: Frame espacial aplanado
            
        Returns:
            (índices_seleccionados, valores_seleccionados)
        """
        # Red neuronal simple para indexing rápido
        hidden = np.tanh(np.dot(spatial_frame, self.indexer_weights) + self.indexer_bias)
        importance = np.dot(hidden, self.output_weights)
        
        # Aplicar softmax para obtener probabilidades
        importance_exp = np.exp(importance - np.max(importance))
        importance_probs = importance_exp / np.sum(importance_exp)
        
        # Seleccionar top-k basado en probabilidades
        top_k_indices = np.argsort(importance_probs)[-self.top_k:]
        selected_values = spatial_frame[top_k_indices]
        
        return top_k_indices, selected_values
    
    def sparse_attention(self, sequence: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Aplica atención dispersa a una secuencia completa.
        
        Args:
            sequence: Secuencia de frames espaciales (seq_len, spatial_dim)
            
        Returns:
            (secuencia_comprimida, lista_de_índices)
        """
        compressed_sequence = []
        selected_indices_list = []
        
        for t in range(sequence.shape[0]):
            frame = sequence[t]
            
            # Aplicar indexador relámpago
            indices, values = self.lightning_indexer(frame)
            
            compressed_sequence.append(values)
            selected_indices_list.append(indices)
        
        return np.array(compressed_sequence), selected_indices_list


class SpatioTemporalTransformer(BaseEstimator, RegressorMixin):
    """
    Modelo principal que combina embeddings espaciales, temporales y atención dispersa
    para predicción meteorológica.
    """
    
    def __init__(self,
                 spatial_shape: Tuple[int, int],
                 sequence_length: int = 48,
                 embedding_dim: int = 64,
                 top_k_ratio: float = 0.1,
                 attention_heads: int = 8,
                 random_state: int = 42):
        """
        Args:
            spatial_shape: Forma espacial original (height, width)
            sequence_length: Longitud de secuencia temporal
            embedding_dim: Dimensión de embeddings
            top_k_ratio: Proporción de píxeles relevantes
            attention_heads: Número de cabezas de atención
        """
        self.spatial_shape = spatial_shape
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.top_k_ratio = top_k_ratio
        self.attention_heads = attention_heads
        self.random_state = random_state
        
        # Calcular dimensión espacial
        self.spatial_dim = np.prod(spatial_shape)
        
        # Inicializar componentes
        self.spatial_embedding = SpatialEmbedding(embedding_dim)
        self.temporal_embedding = TemporalEmbedding(embedding_dim, sequence_length)
        self.sparse_indexer = SparseAttentionIndexer(
            self.spatial_dim, top_k_ratio, attention_heads, random_state
        )
        
        # Modelo de predicción (usando sklearn como backbone)
        self.predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Metadatos del modelo
        self.is_fitted = False
        self.training_stats = {}
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepara características combinando atención dispersa y embeddings.
        
        Args:
            X: Secuencias de entrada (n_samples, sequence_length, spatial_dim)
            
        Returns:
            Características preparadas para el predictor
        """
        n_samples = X.shape[0]
        
        # Crear embeddings temporales una sola vez
        temporal_emb = self.temporal_embedding.create_sequence_embeddings(self.sequence_length)
        
        all_features = []
        
        for i in range(n_samples):
            sequence = X[i]  # (sequence_length, spatial_dim)
            
            # Aplicar atención dispersa
            compressed_seq, indices_list = self.sparse_indexer.sparse_attention(sequence)
            
            # Características temporales: estadísticas sobre la secuencia comprimida
            temporal_features = np.array([
                np.mean(compressed_seq, axis=0),  # Media temporal
                np.std(compressed_seq, axis=0),   # Desviación estándar
                np.max(compressed_seq, axis=0),   # Máximo
                np.min(compressed_seq, axis=0),   # Mínimo
                compressed_seq[-1],               # Último valor
                compressed_seq[0]                 # Primer valor
            ]).flatten()
            
            # Características espaciales: información sobre la distribución de píxeles seleccionados
            spatial_features = []
            for indices in indices_list:
                # Estadísticas sobre los índices seleccionados
                spatial_features.extend([
                    np.mean(indices),
                    np.std(indices),
                    len(np.unique(indices)) / len(indices)  # Diversidad de selección
                ])
            
            spatial_features = np.array(spatial_features)
            
            # Combinar todas las características
            combined_features = np.concatenate([temporal_features, spatial_features])
            all_features.append(combined_features)
        
        return np.array(all_features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SpatioTemporalTransformer':
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X: Secuencias de entrada (n_samples, sequence_length, spatial_dim)
            y: Etiquetas de salida (n_samples, spatial_dim)
            
        Returns:
            self
        """
        print(f"Entrenando modelo con {X.shape[0]} muestras...")
        
        # Preparar características
        X_features = self._prepare_features(X)
        
        # Para la predicción, usaremos la media espacial como objetivo simplificado
        # En un modelo completo, se podría predecir cada píxel individualmente
        y_simplified = np.mean(y, axis=1) if y.ndim > 1 else y
        
        # Entrenar predictor
        self.predictor.fit(X_features, y_simplified)
        
        # Calcular estadísticas de entrenamiento
        y_pred_train = self.predictor.predict(X_features)
        
        self.training_stats = {
            'mse': mean_squared_error(y_simplified, y_pred_train),
            'mae': mean_absolute_error(y_simplified, y_pred_train),
            'r2': r2_score(y_simplified, y_pred_train),
            'n_features': X_features.shape[1],
            'n_samples': X.shape[0]
        }
        
        self.is_fitted = True
        
        print(f"Entrenamiento completado:")
        print(f"  MSE: {self.training_stats['mse']:.4f}")
        print(f"  MAE: {self.training_stats['mae']:.4f}")
        print(f"  R²: {self.training_stats['r2']:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Secuencias de entrada (n_samples, sequence_length, spatial_dim)
            
        Returns:
            Predicciones (n_samples,) - media espacial simplificada
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Preparar características
        X_features = self._prepare_features(X)
        
        # Realizar predicción
        predictions = self.predictor.predict(X_features)
        
        return predictions
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Guarda el modelo completo con metadatos.
        
        Args:
            filepath: Ruta del archivo donde guardar
            metadata: Metadatos adicionales para incluir
        """
        # Preparar datos para guardar
        model_data = {
            'spatial_shape': self.spatial_shape,
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'top_k_ratio': self.top_k_ratio,
            'attention_heads': self.attention_heads,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'training_stats': getattr(self, 'training_stats', None)
        }
        
        # Incluir componentes entrenados si existen
        if hasattr(self, 'predictor') and self.predictor is not None:
            model_data['predictor'] = self.predictor
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            model_data['scaler'] = self.scaler
        
        if hasattr(self, 'spatial_embedding') and self.spatial_embedding is not None:
            model_data['spatial_embedding'] = self.spatial_embedding
        
        if hasattr(self, 'temporal_embedding') and self.temporal_embedding is not None:
            model_data['temporal_embedding'] = self.temporal_embedding
        
        if hasattr(self, 'sparse_indexer') and self.sparse_indexer is not None:
            model_data['sparse_indexer'] = self.sparse_indexer
        
        # Metadatos del modelo
        checkpoint_info = {
            'model_type': 'SpatioTemporalTransformer',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Paquete completo
        complete_package = {
            'checkpoint': checkpoint_info,
            'model_data': model_data
        }
        
        # Guardar archivo
        with open(filepath, 'wb') as f:
            pickle.dump(complete_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Modelo guardado en: {filepath}")
        print(f"Tamaño del archivo: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SpatioTemporalTransformer':
        """
        Carga modelo desde archivo.
        
        Args:
            filepath: Ruta del archivo del modelo
            
        Returns:
            Instancia del modelo cargado
        """
        with open(filepath, 'rb') as f:
            package = pickle.load(f)
        
        # Extraer datos
        checkpoint = package['checkpoint']
        model_data = package['model_data']
        
        # Crear nueva instancia
        model = cls(
            spatial_shape=model_data['spatial_shape'],
            sequence_length=model_data['sequence_length'],
            embedding_dim=model_data['embedding_dim'],
            top_k_ratio=model_data['top_k_ratio'],
            attention_heads=model_data['attention_heads'],
            random_state=model_data['random_state']
        )
        
        # Restaurar estado
        model.is_fitted = model_data['is_fitted']
        
        if model_data.get('training_stats'):
            model.training_stats = model_data['training_stats']
        
        # Restaurar componentes entrenados
        if 'predictor' in model_data:
            model.predictor = model_data['predictor']
        
        if 'scaler' in model_data:
            model.scaler = model_data['scaler']
        
        if 'spatial_embedding' in model_data:
            model.spatial_embedding = model_data['spatial_embedding']
        
        if 'temporal_embedding' in model_data:
            model.temporal_embedding = model_data['temporal_embedding']
        
        if 'sparse_indexer' in model_data:
            model.sparse_indexer = model_data['sparse_indexer']
        
        print(f"Modelo cargado desde: {filepath}")
        print(f"Tipo: {checkpoint['model_type']}, Versión: {checkpoint['version']}")
        print(f"Guardado: {checkpoint['timestamp']}")
        
        return model
    
    def export_to_safetensors(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Exporta el modelo a formato SafeTensors.
        
        Args:
            filepath: Ruta del archivo .safetensors
            metadata: Metadatos adicionales
        """
        try:
            from safetensors.numpy import save_file as safetensors_save
        except ImportError:
            raise ImportError("SafeTensors no disponible. Instalar con: pip install safetensors")
        
        # Extraer arrays numpy del modelo
        tensors = {}
        
        if hasattr(self, 'spatial_embedding') and self.spatial_embedding is not None:
            if hasattr(self.spatial_embedding, 'embeddings'):
                tensors['spatial_embeddings'] = self.spatial_embedding.embeddings.astype(np.float32)
        
        if hasattr(self, 'temporal_embedding') and self.temporal_embedding is not None:
            if hasattr(self.temporal_embedding, 'embeddings'):
                tensors['temporal_embeddings'] = self.temporal_embedding.embeddings.astype(np.float32)
        
        if hasattr(self, 'sparse_indexer') and self.sparse_indexer is not None:
            if hasattr(self.sparse_indexer, 'query_weights'):
                tensors['query_weights'] = self.sparse_indexer.query_weights.astype(np.float32)
            if hasattr(self.sparse_indexer, 'key_weights'):
                tensors['key_weights'] = self.sparse_indexer.key_weights.astype(np.float32)
            if hasattr(self.sparse_indexer, 'indexer_weights'):
                tensors['indexer_weights'] = self.sparse_indexer.indexer_weights.astype(np.float32)
            if hasattr(self.sparse_indexer, 'indexer_bias'):
                tensors['indexer_bias'] = self.sparse_indexer.indexer_bias.astype(np.float32)
            if hasattr(self.sparse_indexer, 'output_weights'):
                tensors['output_weights'] = self.sparse_indexer.output_weights.astype(np.float32)
        
        # Preparar metadatos para SafeTensors
        safe_metadata = {
            'model_type': 'SpatioTemporalTransformer',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'spatial_shape': str(self.spatial_shape),
            'sequence_length': str(self.sequence_length),
            'embedding_dim': str(self.embedding_dim),
            'top_k_ratio': str(self.top_k_ratio),
            'is_fitted': str(self.is_fitted)
        }
        
        if metadata:
            for k, v in metadata.items():
                safe_metadata[f'custom_{k}'] = str(v)
        
        # Guardar con SafeTensors
        safetensors_save(tensors, filepath, metadata=safe_metadata)
        
        print(f"Modelo exportado a SafeTensors: {filepath}")
        print(f"Arrays exportados: {list(tensors.keys())}")
        print(f"Tamaño: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el score R² del modelo.
        
        Args:
            X: Secuencias de entrada
            y: Etiquetas verdaderas
            
        Returns:
            Score R²
        """
        y_pred = self.predict(X)
        y_true = np.mean(y, axis=1) if y.ndim > 1 else y
        
        return r2_score(y_true, y_pred)
    
    @classmethod
    def load_model_old(cls, filepath: str) -> 'SpatioTemporalTransformer':
        """
        Carga un modelo guardado.
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Modelo cargado
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Crear instancia del modelo
        model = cls(
            spatial_shape=model_data['spatial_shape'],
            sequence_length=model_data['sequence_length'],
            embedding_dim=model_data['embedding_dim'],
            top_k_ratio=model_data['top_k_ratio'],
            attention_heads=model_data['attention_heads'],
            random_state=model_data['random_state']
        )
        
        # Restaurar componentes entrenados
        model.predictor = model_data['predictor']
        model.sparse_indexer = model_data['sparse_indexer']
        model.training_stats = model_data['training_stats']
        model.is_fitted = model_data['is_fitted']
        
        print(f"Modelo cargado desde {filepath}")
        return model


def test_model_components():
    """Función de prueba para verificar los componentes del modelo."""
    
    print("=== PROBANDO COMPONENTES DEL MODELO ===")
    
    # Parámetros de prueba
    height, width = 50, 50
    spatial_dim = height * width
    sequence_length = 24
    n_samples = 100
    
    # 1. Probar SpatialEmbedding
    print("\n1. Probando SpatialEmbedding...")
    spatial_emb = SpatialEmbedding(embedding_dim=64)
    spatial_grid = spatial_emb.create_spatial_grid(height, width)
    print(f"   Grilla espacial: {spatial_grid.shape}")
    
    # 2. Probar TemporalEmbedding
    print("\n2. Probando TemporalEmbedding...")
    temporal_emb = TemporalEmbedding(embedding_dim=64)
    temporal_seq = temporal_emb.create_sequence_embeddings(sequence_length)
    print(f"   Secuencia temporal: {temporal_seq.shape}")
    
    # 3. Probar SparseAttentionIndexer
    print("\n3. Probando SparseAttentionIndexer...")
    indexer = SparseAttentionIndexer(spatial_dim, top_k_ratio=0.1)
    test_frame = np.random.randn(spatial_dim)
    indices, values = indexer.lightning_indexer(test_frame)
    print(f"   Píxeles seleccionados: {len(indices)} de {spatial_dim}")
    print(f"   Reducción: {len(indices)/spatial_dim:.1%}")
    
    # 4. Probar modelo completo con datos sintéticos
    print("\n4. Probando SpatioTemporalTransformer...")
    
    # Crear datos sintéticos
    X_synthetic = np.random.randn(n_samples, sequence_length, spatial_dim)
    y_synthetic = np.random.randn(n_samples, spatial_dim)
    
    # Crear y entrenar modelo
    model = SpatioTemporalTransformer(
        spatial_shape=(height, width),
        sequence_length=sequence_length,
        top_k_ratio=0.05  # Más agresivo para prueba
    )
    
    # Dividir en train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_synthetic[:split_idx], X_synthetic[split_idx:]
    y_train, y_test = y_synthetic[:split_idx], y_synthetic[split_idx:]
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"   Score de entrenamiento: {train_score:.4f}")
    print(f"   Score de prueba: {test_score:.4f}")
    
    # Hacer algunas predicciones
    predictions = model.predict(X_test[:5])
    print(f"   Predicciones de muestra: {predictions[:3]}")
    
    print("\n=== PRUEBAS COMPLETADAS ===")


if __name__ == "__main__":
    test_model_components()