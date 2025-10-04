#!/usr/bin/env python3
"""
Script Principal de Entrenamiento del Modelo de Predicción Meteorológica
======================================================================

Este script integra todos los componentes del sistema de predicción meteorológica
inspirado en DeepSeek's Sparse Attention para entrenar un modelo completo.

Funcionalidades:
1. Carga de datos preprocesados
2. Configuración del modelo con atención dispersa
3. Entrenamiento con validación cruzada
4. Evaluación y métricas de rendimiento
5. Visualización de resultados
6. Guardado del modelo entrenado
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos personalizados
from data_preprocessing import EarthDataPreprocessor
from spatio_temporal_transformer import SpatioTemporalTransformer
from geospatial_sparse_attention import GeospatialSparseAttention, visualize_attention_mask

# Métricas de evaluación
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import TimeSeriesSplit


class WeatherPredictionTrainer:
    """
    Entrenador principal para el modelo de predicción meteorológica.
    """
    
    def __init__(self, 
                 config: Dict = None,
                 data_dir: str = "processed_data",
                 model_dir: str = "trained_models",
                 results_dir: str = "training_results"):
        """
        Args:
            config: Configuración del modelo
            data_dir: Directorio con datos preprocesados
            model_dir: Directorio para guardar modelos
            results_dir: Directorio para resultados de entrenamiento
        """
        self.config = config or self._default_config()
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        # Crear directorios si no existen
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Placeholders para datos y modelo
        self.data_splits = None
        self.metadata = None
        self.model = None
        self.training_history = []
        
        # Configurar logging
        self.setup_logging()
    
    def _default_config(self) -> Dict:
        """Configuración por defecto del modelo."""
        return {
            'model': {
                'embedding_dim': 64,
                'top_k_ratio': 0.1,
                'attention_heads': 8,
                'sequence_length': 24,
                'random_state': 42
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'cv_folds': 3,
                'random_state': 42
            },
            'sparse_attention': {
                'sparsity_ratio': 0.1,
                'weather_aware': True,
                'multi_scale': True,
                'adaptive_threshold': True
            }
        }
    
    def setup_logging(self):
        """Configura el sistema de logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"weather_prediction_{timestamp}"
        self.log_file = os.path.join(self.results_dir, f"{self.experiment_id}.log")
        
        print(f"Experimento ID: {self.experiment_id}")
        print(f"Log file: {self.log_file}")
    
    def log_message(self, message: str, print_msg: bool = True):
        """
        Registra un mensaje en el log y opcionalmente lo imprime.
        
        Args:
            message: Mensaje a registrar
            print_msg: Si imprimir el mensaje en consola
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
        
        if print_msg:
            print(message)
    
    def load_processed_data(self) -> None:
        """Carga los datos preprocesados."""
        self.log_message("=== CARGANDO DATOS PREPROCESADOS ===")
        
        try:
            # Cargar splits de datos
            data_files = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            self.data_splits = {}
            
            for file_name in data_files:
                file_path = os.path.join(self.data_dir, 'sequences', f'{file_name}.npy')
                if os.path.exists(file_path):
                    self.data_splits[file_name] = np.load(file_path)
                    self.log_message(f"Cargado {file_name}: {self.data_splits[file_name].shape}")
                else:
                    raise FileNotFoundError(f"No se encontró {file_path}")
            
            # Cargar metadatos
            metadata_path = os.path.join(self.data_dir, 'metadata', 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.log_message(f"Metadatos cargados: {len(self.metadata)} campos")
            else:
                self.log_message("Advertencia: No se encontraron metadatos")
                self.metadata = {}
            
            # Verificar integridad de los datos
            self._verify_data_integrity()
            
        except Exception as e:
            self.log_message(f"Error cargando datos: {e}")
            raise
    
    def _verify_data_integrity(self) -> None:
        """Verifica la integridad de los datos cargados."""
        self.log_message("Verificando integridad de datos...")
        
        # Verificar formas consistentes
        n_train = len(self.data_splits['X_train'])
        n_val = len(self.data_splits['X_val'])
        n_test = len(self.data_splits['X_test'])
        
        assert len(self.data_splits['y_train']) == n_train, "Inconsistencia en tamaño de entrenamiento"
        assert len(self.data_splits['y_val']) == n_val, "Inconsistencia en tamaño de validación"
        assert len(self.data_splits['y_test']) == n_test, "Inconsistencia en tamaño de prueba"
        
        # Verificar que no hay NaN
        for name, data in self.data_splits.items():
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                self.log_message(f"Advertencia: {name} contiene {nan_count} valores NaN")
        
        # Estadísticas de los datos
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        
        self.log_message(f"Estadísticas de datos:")
        self.log_message(f"  X_train - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}, Std: {X_train.std():.4f}")
        self.log_message(f"  y_train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Std: {y_train.std():.4f}")
        
        self.log_message("Verificación de integridad completada ✓")
    
    def create_model(self) -> None:
        """Crea el modelo con la configuración especificada."""
        self.log_message("=== CREANDO MODELO ===")
        
        # Extraer configuración
        model_config = self.config['model']
        
        # Determinar forma espacial de los metadatos o datos
        y_shape = self.data_splits['y_train'].shape
        if len(y_shape) == 2:
            # Calcular forma espacial basada en los datos reales
            spatial_dim = y_shape[1]  # 32400
            
            if 'spatial_shape' in self.metadata:
                metadata_shape = tuple(self.metadata['spatial_shape'])
                # Verificar si la forma de metadatos coincide con los datos
                if len(metadata_shape) >= 2:
                    if len(metadata_shape) == 3:  # (canales, altura, anchura)
                        expected_dim = metadata_shape[1] * metadata_shape[2]
                    else:  # (altura, anchura)
                        expected_dim = metadata_shape[0] * metadata_shape[1]
                    
                    if expected_dim == spatial_dim:
                        spatial_shape = metadata_shape[-2:]  # Tomar altura, anchura
                    else:
                        # Forma de metadatos no coincide, inferir
                        side_length = int(np.sqrt(spatial_dim))
                        spatial_shape = (side_length, side_length)
                else:
                    side_length = int(np.sqrt(spatial_dim))
                    spatial_shape = (side_length, side_length)
            else:
                # Inferir forma cuadrada
                side_length = int(np.sqrt(spatial_dim))
                spatial_shape = (side_length, side_length)
        else:
            spatial_shape = (50, 50)  # Fallback
        
        self.log_message(f"Forma espacial inferida: {spatial_shape}")
        
        # Crear modelo
        self.model = SpatioTemporalTransformer(
            spatial_shape=spatial_shape,
            sequence_length=model_config['sequence_length'],
            embedding_dim=model_config['embedding_dim'],
            top_k_ratio=model_config['top_k_ratio'],
            attention_heads=model_config['attention_heads'],
            random_state=model_config['random_state']
        )
        
        self.log_message(f"Modelo creado con configuración:")
        for key, value in model_config.items():
            self.log_message(f"  {key}: {value}")
    
    def train_model(self) -> None:
        """Entrena el modelo principal."""
        self.log_message("=== ENTRENANDO MODELO ===")
        
        if self.model is None:
            raise ValueError("Modelo no creado. Ejecutar create_model() primero.")
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        X_val = self.data_splits['X_val']
        y_val = self.data_splits['y_val']
        
        start_time = datetime.now()
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        self.log_message(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Evaluar en conjuntos de entrenamiento y validación
        train_metrics = self._evaluate_model(X_train, y_train, "Entrenamiento")
        val_metrics = self._evaluate_model(X_val, y_val, "Validación")
        
        # Guardar métricas
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config
        })
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, 
                       dataset_name: str) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            X: Datos de entrada
            y: Etiquetas verdaderas
            dataset_name: Nombre del conjunto de datos
            
        Returns:
            Diccionario con métricas
        """
        predictions = self.model.predict(X)
        
        # Para evaluación, usar la media espacial
        if y.ndim > 1:
            y_eval = np.mean(y, axis=1)
        else:
            y_eval = y
        
        # Calcular métricas
        mse = mean_squared_error(y_eval, predictions)
        mae = mean_absolute_error(y_eval, predictions)
        r2 = r2_score(y_eval, predictions)
        explained_var = explained_variance_score(y_eval, predictions)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # Error relativo promedio
        mean_actual = np.mean(np.abs(y_eval))
        relative_error = mae / mean_actual if mean_actual > 0 else float('inf')
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'relative_error': relative_error
        }
        
        # Log de métricas
        self.log_message(f"Métricas para {dataset_name}:")
        for metric_name, value in metrics.items():
            self.log_message(f"  {metric_name}: {value:.6f}")
        
        return metrics
    
    def cross_validate(self) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada temporal.
        
        Returns:
            Diccionario con métricas de CV
        """
        self.log_message("=== VALIDACIÓN CRUZADA TEMPORAL ===")
        
        X_full = np.vstack([self.data_splits['X_train'], self.data_splits['X_val']])
        y_full = np.vstack([self.data_splits['y_train'], self.data_splits['y_val']])
        
        cv_config = self.config['training']
        n_splits = cv_config['cv_folds']
        
        # Usar TimeSeriesSplit para mantener orden temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = {
            'train_r2': [],
            'val_r2': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
            self.log_message(f"Fold {fold + 1}/{n_splits}")
            
            X_train_fold = X_full[train_idx]
            y_train_fold = y_full[train_idx]
            X_val_fold = X_full[val_idx]
            y_val_fold = y_full[val_idx]
            
            # Crear modelo para este fold
            fold_model = SpatioTemporalTransformer(
                spatial_shape=self.model.spatial_shape,
                sequence_length=self.model.sequence_length,
                embedding_dim=self.model.embedding_dim,
                top_k_ratio=self.model.top_k_ratio,
                attention_heads=self.model.attention_heads,
                random_state=self.config['model']['random_state']
            )
            
            # Entrenar
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluar
            train_metrics = self._evaluate_model(X_train_fold, y_train_fold, f"Fold {fold+1} Train")
            val_metrics = self._evaluate_model(X_val_fold, y_val_fold, f"Fold {fold+1} Val")
            
            # Guardar métricas
            cv_metrics['train_r2'].append(train_metrics['r2'])
            cv_metrics['val_r2'].append(val_metrics['r2'])
            cv_metrics['train_rmse'].append(train_metrics['rmse'])
            cv_metrics['val_rmse'].append(val_metrics['rmse'])
            cv_metrics['train_mae'].append(train_metrics['mae'])
            cv_metrics['val_mae'].append(val_metrics['mae'])
        
        # Estadísticas de CV
        self.log_message("Resultados de Validación Cruzada:")
        for metric_name, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            self.log_message(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
        
        return cv_metrics
    
    def final_evaluation(self) -> Dict[str, float]:
        """
        Evaluación final en el conjunto de prueba.
        
        Returns:
            Métricas finales
        """
        self.log_message("=== EVALUACIÓN FINAL ===")
        
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        
        final_metrics = self._evaluate_model(X_test, y_test, "Prueba Final")
        
        return final_metrics
    
    def visualize_results(self) -> None:
        """Crea visualizaciones de los resultados del entrenamiento."""
        self.log_message("=== CREANDO VISUALIZACIONES ===")
        
        # 1. Predicciones vs Valores Reales
        self._plot_predictions_vs_actual()
        
        # 2. Distribución de errores
        self._plot_error_distribution()
        
        # 3. Análisis de atención dispersa
        self._analyze_sparse_attention()
        
        self.log_message("Visualizaciones guardadas en " + self.results_dir)
    
    def _plot_predictions_vs_actual(self) -> None:
        """Gráfico de predicciones vs valores reales."""
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        
        predictions = self.model.predict(X_test)
        y_actual = np.mean(y_test, axis=1) if y_test.ndim > 1 else y_test
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_actual, predictions, alpha=0.6)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        
        # Residuos
        plt.subplot(2, 2, 2)
        residuals = y_actual - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Gráfico de Residuos')
        
        # Serie temporal de predicciones
        plt.subplot(2, 2, 3)
        n_show = min(100, len(y_actual))
        plt.plot(y_actual[:n_show], label='Real', alpha=0.8)
        plt.plot(predictions[:n_show], label='Predicción', alpha=0.8)
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.title('Serie Temporal (Primeras 100 muestras)')
        plt.legend()
        
        # Histograma de errores
        plt.subplot(2, 2, 4)
        n_bins = min(10, max(2, len(residuals)//2))  # Asegurar bins apropiados
        plt.hist(residuals, bins=n_bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Error (Real - Predicción)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.experiment_id}_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self) -> None:
        """Análisis detallado de la distribución de errores."""
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        
        predictions = self.model.predict(X_test)
        y_actual = np.mean(y_test, axis=1) if y_test.ndim > 1 else y_test
        
        errors = np.abs(y_actual - predictions)
        relative_errors = errors / (np.abs(y_actual) + 1e-8)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribución de errores absolutos
        n_bins = min(20, max(2, len(errors)//2))
        axes[0, 0].hist(errors, bins=n_bins, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Error Absoluto')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].set_title('Distribución de Errores Absolutos')
        
        # Distribución de errores relativos
        axes[0, 1].hist(relative_errors, bins=n_bins, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Error Relativo')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Errores Relativos')
        
        # Errores vs magnitud de la predicción
        axes[1, 0].scatter(np.abs(y_actual), errors, alpha=0.6)
        axes[1, 0].set_xlabel('Magnitud del Valor Real')
        axes[1, 0].set_ylabel('Error Absoluto')
        axes[1, 0].set_title('Error vs Magnitud')
        
        # Box plot de errores por cuartiles (solo si hay suficientes datos únicos)
        try:
            if len(np.unique(y_actual)) >= 4:
                quartiles = pd.qcut(y_actual, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                error_df = pd.DataFrame({'Error': errors, 'Cuartil': quartiles})
                sns.boxplot(data=error_df, x='Cuartil', y='Error', ax=axes[1, 1])
                axes[1, 1].set_title('Errores por Cuartiles de Magnitud')
            else:
                # Gráfico alternativo para pocos datos
                axes[1, 1].hist(errors, bins=max(2, len(errors)//2), alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Error Absoluto')
                axes[1, 1].set_ylabel('Frecuencia')
                axes[1, 1].set_title('Distribución de Errores (Pocos Datos)')
        except Exception as e:
            # Fallback simple
            axes[1, 1].plot(range(len(errors)), errors, 'o-')
            axes[1, 1].set_xlabel('Muestra')
            axes[1, 1].set_ylabel('Error')
            axes[1, 1].set_title('Errores por Muestra')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{self.experiment_id}_error_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_sparse_attention(self) -> None:
        """Analiza el comportamiento de la atención dispersa."""
        # Tomar una muestra de datos de prueba
        X_sample = self.data_splits['X_test'][:5]  # Primeras 5 muestras
        
        # Crear instancia de atención dispersa para análisis
        sparse_attention = GeospatialSparseAttention(
            spatial_shape=self.model.spatial_shape,
            sparsity_ratio=self.config['sparse_attention']['sparsity_ratio'],
            weather_aware=self.config['sparse_attention']['weather_aware'],
            multi_scale=self.config['sparse_attention']['multi_scale'],
            adaptive_threshold=self.config['sparse_attention']['adaptive_threshold']
        )
        
        # Analizar patrones de atención para cada muestra
        for i, sample in enumerate(X_sample):
            # Tomar el último frame temporal de la secuencia
            last_frame = sample[-1].reshape(self.model.spatial_shape)
            
            # Aplicar atención dispersa
            attention_mask = sparse_attention.lightning_indexer(
                last_frame,
                frame_type='precipitation'
            )
            
            # Visualizar
            save_path = os.path.join(self.results_dir, 
                                   f'{self.experiment_id}_attention_sample_{i+1}.png')
            visualize_attention_mask(attention_mask, last_frame, save_path)
    
    def save_model_and_results(self) -> None:
        """Guarda el modelo entrenado y los resultados en múltiples formatos."""
        self.log_message("=== GUARDANDO MODELO Y RESULTADOS ===")
        
        # Metadatos del modelo
        model_metadata = {
            'experiment_id': self.experiment_id,
            'training_samples': len(self.data_splits['X_train']),
            'validation_samples': len(self.data_splits['X_val']),
            'test_samples': len(self.data_splits['X_test']),
            'config': self.config,
            'final_metrics': getattr(self, 'final_metrics', {}),
            'training_time': getattr(self, 'training_time', 0),
            'model_architecture': {
                'spatial_shape': self.model.spatial_shape,
                'sequence_length': self.model.sequence_length,
                'embedding_dim': self.model.embedding_dim,
                'top_k_ratio': self.model.top_k_ratio
            }
        }
        
        # 1. Guardar en formato Pickle (principal)
        model_path_pkl = os.path.join(self.model_dir, f'{self.experiment_id}_model.pkl')
        self.model.save_model(model_path_pkl, metadata=model_metadata)
        self.log_message(f"✓ Modelo guardado (Pickle): {model_path_pkl}")
        
        # 2. Intentar guardar en SafeTensors
        try:
            model_path_safetensors = os.path.join(self.model_dir, f'{self.experiment_id}_model.safetensors')
            self.model.export_to_safetensors(model_path_safetensors, metadata=model_metadata)
            self.log_message(f"✓ Modelo exportado (SafeTensors): {model_path_safetensors}")
        except ImportError:
            self.log_message("⚠ SafeTensors no disponible. Instalar con: pip install safetensors")
        except Exception as e:
            self.log_message(f"⚠ Error exportando SafeTensors: {e}")
        
        # 3. Guardar modelo de persistencia avanzada (si disponible)
        try:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from model_persistence import ModelPersistence
            
            persistence = ModelPersistence(compression=True, validate_integrity=True)
            
            # HDF5 format
            try:
                model_path_h5 = os.path.join(self.model_dir, f'{self.experiment_id}_model.h5')
                persistence.save_model_hdf5(self.model, model_path_h5, metadata=model_metadata)
                self.log_message(f"✓ Modelo guardado (HDF5): {model_path_h5}")
            except ImportError:
                self.log_message("⚠ h5py no disponible para formato HDF5")
            except Exception as e:
                self.log_message(f"⚠ Error guardando HDF5: {e}")
        
        except ImportError:
            self.log_message("⚠ model_persistence.py no encontrado")
        
        # 4. Guardar historial de entrenamiento
        history_path = os.path.join(self.results_dir, f'{self.experiment_id}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        self.log_message(f"✓ Historial guardado: {history_path}")
        
        # 5. Guardar configuración completa
        config_path = os.path.join(self.results_dir, f'{self.experiment_id}_config.json')
        complete_config = {
            'training_config': self.config,
            'model_metadata': model_metadata,
            'available_formats': self._check_available_formats()
        }
        with open(config_path, 'w') as f:
            json.dump(complete_config, f, indent=2, default=str)
        self.log_message(f"✓ Configuración guardada: {config_path}")
        
        # 6. Crear script de carga de ejemplo
        loader_script = self._create_model_loader_script()
        loader_path = os.path.join(self.model_dir, f'{self.experiment_id}_load_model.py')
        with open(loader_path, 'w') as f:
            f.write(loader_script)
        self.log_message(f"✓ Script de carga creado: {loader_path}")
    
    def _check_available_formats(self) -> Dict[str, bool]:
        """Verifica qué formatos de guardado están disponibles."""
        formats = {
            'pickle': True,  # Siempre disponible
            'safetensors': False,
            'hdf5': False,
            'json_numpy': True  # Usando numpy y json estándar
        }
        
        try:
            import safetensors
            formats['safetensors'] = True
        except ImportError:
            pass
        
        try:
            import h5py
            formats['hdf5'] = True
        except ImportError:
            pass
        
        return formats
    
    def _create_model_loader_script(self) -> str:
        """Crea script de ejemplo para cargar el modelo."""
        script = f'''#!/usr/bin/env python3
"""
Script de Ejemplo para Cargar Modelo Entrenado
==============================================

Modelo: {self.experiment_id}
Generado automáticamente por el sistema de entrenamiento.
"""

import os
import sys
import numpy as np

# Añadir directorio de scripts al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def load_model_pickle():
    """Carga modelo desde formato Pickle."""
    from spatio_temporal_transformer import SpatioTemporalTransformer
    
    model_path = "{self.experiment_id}_model.pkl"
    model = SpatioTemporalTransformer.load_model(model_path)
    
    print(f"Modelo cargado desde Pickle: {{model_path}}")
    print(f"Configuración del modelo:")
    print(f"  - Forma espacial: {{model.spatial_shape}}")
    print(f"  - Longitud de secuencia: {{model.sequence_length}}")
    print(f"  - Embedding dim: {{model.embedding_dim}}")
    print(f"  - Top-k ratio: {{model.top_k_ratio}}")
    print(f"  - Entrenado: {{model.is_fitted}}")
    
    return model

def load_model_safetensors():
    """Carga tensores desde SafeTensors (solo arrays numpy)."""
    try:
        from safetensors import safe_open
        
        tensors_path = "{self.experiment_id}_model.safetensors"
        
        tensors = {{}}
        with safe_open(tensors_path, framework="numpy") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        print(f"Tensores cargados desde SafeTensors: {{tensors_path}}")
        print(f"Arrays disponibles: {{list(tensors.keys())}}")
        print(f"Metadatos: {{metadata}}")
        
        return tensors, metadata
    
    except ImportError:
        print("SafeTensors no disponible. Instalar con: pip install safetensors")
        return None, None

def test_model_predictions():
    """Prueba las predicciones del modelo."""
    model = load_model_pickle()
    
    if model.is_fitted:
        # Crear datos de prueba sintéticos
        spatial_dim = np.prod(model.spatial_shape)
        test_sequence = np.random.randn(5, model.sequence_length, spatial_dim)
        
        print("\\nProbando predicciones...")
        predictions = model.predict(test_sequence)
        
        print(f"Entrada: {{test_sequence.shape}}")
        print(f"Predicciones: {{predictions.shape}}")
        print(f"Valores de ejemplo: {{predictions[:3]}}")
    else:
        print("El modelo no está entrenado.")

def main():
    """Función principal de demostración."""
    print("=" * 50)
    print(f"CARGANDO MODELO: {self.experiment_id}")
    print("=" * 50)
    
    # Cargar y probar modelo principal
    model = load_model_pickle()
    
    # Cargar tensores SafeTensors si está disponible
    tensors, metadata = load_model_safetensors()
    
    # Probar predicciones
    test_model_predictions()
    
    print("\\n✓ Carga de modelo completada exitosamente")

if __name__ == "__main__":
    main()
'''
        return script
    
    def run_complete_training(self) -> None:
        """Ejecuta el pipeline completo de entrenamiento."""
        try:
            self.log_message("=== INICIANDO ENTRENAMIENTO COMPLETO ===")
            
            # 1. Cargar datos
            self.load_processed_data()
            
            # 2. Crear modelo
            self.create_model()
            
            # 3. Entrenar modelo
            self.train_model()
            
            # 4. Validación cruzada
            cv_results = self.cross_validate()
            
            # 5. Evaluación final
            final_metrics = self.final_evaluation()
            
            # 6. Visualizaciones
            self.visualize_results()
            
            # 7. Guardar resultados
            self.save_model_and_results()
            
            self.log_message("=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
            
            # Resumen final
            self.log_message("RESUMEN FINAL:")
            self.log_message(f"  R² final: {final_metrics['r2']:.4f}")
            self.log_message(f"  RMSE final: {final_metrics['rmse']:.4f}")
            self.log_message(f"  MAE final: {final_metrics['mae']:.4f}")
            
        except Exception as e:
            self.log_message(f"Error durante el entrenamiento: {e}")
            raise


def main():
    """Función principal."""
    print("=== ENTRENADOR DE MODELO DE PREDICCIÓN METEOROLÓGICA ===")
    print("Inspirado en DeepSeek's Sparse Attention para datos geoespaciales")
    
    # Verificar que existen datos preprocesados
    if not os.path.exists("processed_data"):
        print("\nError: No se encontraron datos preprocesados.")
        print("Ejecuta primero el script de preprocesamiento:")
        print("python3 scripts/data_preprocessing.py")
        return
    
    # Configuración personalizada (opcional)
    custom_config = {
        'model': {
            'embedding_dim': 64,
            'top_k_ratio': 0.15,  # Un poco más agresivo
            'attention_heads': 6,
            'sequence_length': 24,
            'random_state': 42
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.2,
            'cv_folds': 3,
            'random_state': 42
        },
        'sparse_attention': {
            'sparsity_ratio': 0.12,
            'weather_aware': True,
            'multi_scale': True,
            'adaptive_threshold': True
        }
    }
    
    # Crear entrenador
    trainer = WeatherPredictionTrainer(
        config=custom_config,
        data_dir="processed_data",
        model_dir="trained_models",
        results_dir="training_results"
    )
    
    # Ejecutar entrenamiento completo
    trainer.run_complete_training()
    
    print(f"\nEntrenamiento completado. Revisa los resultados en:")
    print(f"  - Modelos: {trainer.model_dir}")
    print(f"  - Resultados: {trainer.results_dir}")
    print(f"  - Log: {trainer.log_file}")


if __name__ == "__main__":
    main()