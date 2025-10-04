#!/usr/bin/env python3
"""
Entrenamiento con Dataset Grande
===============================

Script optimizado para entrenar modelos con grandes volúmenes de datos
usando carga por lotes, entrenamiento incremental y monitoreo avanzado.
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import psutil
import gc

# Importar componentes del modelo
sys.path.append(str(Path(__file__).parent))
from spatio_temporal_transformer import SpatioTemporalTransformer
from model_persistence import ModelPersistence

class LargeDatasetTrainer:
    """Entrenador optimizado para datasets grandes."""
    
    def __init__(self,
                 processed_data_dir="processed_data_large",
                 model_dir="trained_models_large",
                 batch_load_size=5,
                 validation_split=0.15,
                 test_split=0.15,
                 incremental_training=True):
        """
        Inicializar entrenador para datasets grandes.
        
        Args:
            processed_data_dir: Directorio con datos procesados
            model_dir: Directorio para modelos entrenados
            batch_load_size: Lotes de datos a cargar en memoria
            validation_split: Proporción para validación
            test_split: Proporción para prueba
            incremental_training: Si usar entrenamiento incremental
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.model_dir = Path(model_dir)
        self.batch_load_size = batch_load_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.incremental_training = incremental_training
        
        # Crear directorios
        self.model_dir.mkdir(exist_ok=True)
        (self.model_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Cargar metadatos del preprocesamiento
        self.metadata = self.load_processing_metadata()
        
        # Estado del entrenamiento
        self.model = None
        self.training_history = []
        
        print(f"🏋️ Configuración del entrenador:")
        print(f"   - Datos procesados: {self.processed_data_dir}")
        print(f"   - Lotes en memoria: {batch_load_size}")
        print(f"   - Validación: {validation_split:.1%}")
        print(f"   - Prueba: {test_split:.1%}")
        print(f"   - Entrenamiento incremental: {incremental_training}")
    
    def load_processing_metadata(self):
        """Carga metadatos del preprocesamiento."""
        metadata_file = self.processed_data_dir / "metadata" / "processing_info.json"
        
        if not metadata_file.exists():
            print(f"❌ No se encontraron metadatos en {metadata_file}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📋 Metadatos cargados:")
        print(f"   - Archivos fuente: {metadata['dataset_info']['total_source_files']}")
        print(f"   - Secuencias totales: {metadata['dataset_info']['total_sequences']}")
        print(f"   - Forma espacial: {metadata['dataset_info']['spatial_shape']}")
        
        return metadata
    
    def discover_batch_files(self):
        """Descubre archivos de lotes procesados."""
        batch_dir = self.processed_data_dir / "batches"
        batch_files = list(batch_dir.glob("batch_*.npz"))
        
        if not batch_files:
            print(f"❌ No se encontraron lotes en {batch_dir}")
            return []
        
        # Ordenar por índice de lote
        batch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        print(f"📦 Encontrados {len(batch_files)} lotes de datos")
        return batch_files
    
    def load_batch_data(self, batch_files):
        """
        Carga datos de múltiples lotes.
        
        Args:
            batch_files: Lista de archivos de lote
            
        Returns:
            Tupla (X, y) con datos combinados
        """
        X_batches = []
        y_batches = []
        
        for batch_file in batch_files:
            try:
                print(f"📥 Cargando {batch_file.name}...")
                with np.load(batch_file) as data:
                    X_batch = data['X']
                    y_batch = data['y']
                    
                    X_batches.append(X_batch)
                    y_batches.append(y_batch)
                    
                    print(f"   ✅ {X_batch.shape[0]} secuencias cargadas")
            
            except Exception as e:
                print(f"❌ Error cargando {batch_file.name}: {e}")
        
        if X_batches and y_batches:
            X = np.vstack(X_batches)
            y = np.vstack(y_batches)
            
            print(f"🔗 Datos combinados: X={X.shape}, y={y.shape}")
            return X, y
        else:
            return None, None
    
    def create_data_splits(self, batch_files):
        """
        Crea divisiones de datos para entrenamiento/validación/prueba.
        
        Args:
            batch_files: Lista de archivos de lote
            
        Returns:
            Diccionario con índices de divisiones
        """
        n_batches = len(batch_files)
        
        # Calcular índices de división
        test_batches = max(1, int(n_batches * self.test_split))
        val_batches = max(1, int(n_batches * self.validation_split))
        train_batches = n_batches - test_batches - val_batches
        
        # Asegurar que hay al menos un lote para entrenamiento
        if train_batches < 1:
            train_batches = 1
            val_batches = max(0, n_batches - 2)
            test_batches = max(0, n_batches - train_batches - val_batches)
        
        splits = {
            'train': batch_files[:train_batches],
            'val': batch_files[train_batches:train_batches + val_batches],
            'test': batch_files[train_batches + val_batches:]
        }
        
        print(f"📊 División de datos:")
        print(f"   - Entrenamiento: {len(splits['train'])} lotes")
        print(f"   - Validación: {len(splits['val'])} lotes")
        print(f"   - Prueba: {len(splits['test'])} lotes")
        
        return splits
    
    def create_model(self):
        """Crea modelo basado en metadatos."""
        if not self.metadata:
            print("❌ No hay metadatos para crear el modelo")
            return None
        
        # Extraer configuración
        spatial_shape = tuple(self.metadata['dataset_info']['spatial_shape'])
        sequence_length = self.metadata['processing_config']['sequence_length']
        
        # Configuración del modelo para dataset grande
        model_config = {
            'spatial_shape': spatial_shape,
            'sequence_length': sequence_length,
            'embedding_dim': 128,  # Más grande para datasets grandes
            'top_k_ratio': 0.08,   # Más selectivo
            'attention_heads': 8,   # Más cabezas de atención
            'random_state': 42
        }
        
        print(f"🧠 Creando modelo con configuración:")
        for key, value in model_config.items():
            print(f"   - {key}: {value}")
        
        self.model = SpatioTemporalTransformer(**model_config)
        return self.model
    
    def train_incremental(self, data_splits):
        """
        Entrenamiento incremental por lotes.
        
        Args:
            data_splits: Diccionario con divisiones de datos
        """
        print("\n🔄 INICIANDO ENTRENAMIENTO INCREMENTAL")
        print("=" * 50)
        
        train_batches = data_splits['train']
        val_batches = data_splits['val']
        
        # Cargar datos de validación (más pequeños)
        if val_batches:
            X_val, y_val = self.load_batch_data(val_batches)
        else:
            X_val, y_val = None, None
        
        # Entrenar incrementalmente
        total_samples_trained = 0
        
        for epoch in range(3):  # Múltiples épocas
            print(f"\n--- Época {epoch + 1}/3 ---")
            
            for i in range(0, len(train_batches), self.batch_load_size):
                batch_group = train_batches[i:i + self.batch_load_size]
                
                print(f"\n📚 Entrenando con lotes {i//self.batch_load_size + 1}")
                
                # Cargar grupo de lotes
                X_train, y_train = self.load_batch_data(batch_group)
                
                if X_train is not None and y_train is not None:
                    # Entrenar modelo
                    start_time = time.time()
                    self.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    total_samples_trained += len(X_train)
                    
                    # Evaluar en validación si disponible
                    if X_val is not None and y_val is not None:
                        val_score = self.model.score(X_val, y_val)
                        print(f"   📊 Score validación: R² = {val_score:.4f}")
                    
                    # Estadísticas de entrenamiento
                    train_score = self.model.score(X_train, y_train)
                    
                    stats = {
                        'epoch': epoch + 1,
                        'batch_group': i//self.batch_load_size + 1,
                        'samples_trained': len(X_train),
                        'total_samples': total_samples_trained,
                        'train_r2': train_score,
                        'val_r2': val_score if X_val is not None else None,
                        'training_time': training_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.training_history.append(stats)
                    
                    print(f"   ✅ Entrenamiento: R² = {train_score:.4f}")
                    print(f"   ⏱️ Tiempo: {training_time:.2f}s")
                    print(f"   📈 Total muestras: {total_samples_trained}")
                    
                    # Guardar checkpoint periódico
                    if (i // self.batch_load_size + 1) % 5 == 0:
                        self.save_checkpoint(epoch, i // self.batch_load_size + 1)
                
                # Limpiar memoria
                del X_train, y_train
                gc.collect()
        
        print(f"\n✅ Entrenamiento incremental completado")
        print(f"📊 Total muestras entrenadas: {total_samples_trained}")
    
    def train_batch_loading(self, data_splits):
        """
        Entrenamiento con carga de lotes completos.
        
        Args:
            data_splits: Diccionario con divisiones de datos
        """
        print("\n🏋️ INICIANDO ENTRENAMIENTO CON CARGA POR LOTES")
        print("=" * 50)
        
        # Cargar datos de entrenamiento
        print("📥 Cargando datos de entrenamiento...")
        X_train, y_train = self.load_batch_data(data_splits['train'])
        
        if X_train is None or y_train is None:
            print("❌ No se pudieron cargar datos de entrenamiento")
            return
        
        # Cargar datos de validación
        X_val, y_val = None, None
        if data_splits['val']:
            print("📥 Cargando datos de validación...")
            X_val, y_val = self.load_batch_data(data_splits['val'])
        
        # Entrenar modelo
        print(f"🚀 Entrenando modelo con {len(X_train)} muestras...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluar modelo
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None
        
        # Registrar estadísticas
        stats = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if X_val is not None else 0,
            'train_r2': train_score,
            'val_r2': val_score,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(stats)
        
        print(f"✅ Entrenamiento completado:")
        print(f"   📊 Score entrenamiento: R² = {train_score:.4f}")
        if val_score is not None:
            print(f"   📊 Score validación: R² = {val_score:.4f}")
        print(f"   ⏱️ Tiempo total: {training_time:.2f}s")
    
    def evaluate_final_model(self, data_splits):
        """Evaluación final del modelo en datos de prueba."""
        if not data_splits['test']:
            print("⚠️ No hay datos de prueba para evaluación final")
            return
        
        print("\n🎯 EVALUACIÓN FINAL")
        print("=" * 30)
        
        # Cargar datos de prueba
        print("📥 Cargando datos de prueba...")
        X_test, y_test = self.load_batch_data(data_splits['test'])
        
        if X_test is None or y_test is None:
            print("❌ No se pudieron cargar datos de prueba")
            return
        
        # Evaluar modelo
        test_score = self.model.score(X_test, y_test)
        
        # Hacer predicciones de muestra
        sample_predictions = self.model.predict(X_test[:5])
        
        print(f"📊 Resultados finales:")
        print(f"   - Muestras de prueba: {len(X_test)}")
        print(f"   - Score R²: {test_score:.4f}")
        print(f"   - Predicciones muestra: {sample_predictions.shape}")
        
        # Agregar a historial
        final_stats = {
            'phase': 'final_evaluation',
            'test_samples': len(X_test),
            'test_r2': test_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(final_stats)
        
        return test_score
    
    def save_checkpoint(self, epoch, batch_group):
        """Guarda checkpoint del modelo."""
        checkpoint_name = f"checkpoint_epoch_{epoch}_batch_{batch_group}.pkl"
        checkpoint_path = self.model_dir / "checkpoints" / checkpoint_name
        
        metadata = {
            'epoch': epoch,
            'batch_group': batch_group,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        self.model.save_model(str(checkpoint_path), metadata)
        print(f"💾 Checkpoint guardado: {checkpoint_name}")
    
    def save_final_model(self):
        """Guarda modelo final con todos los formatos."""
        print("\n💾 GUARDANDO MODELO FINAL")
        print("=" * 30)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Metadatos completos
        final_metadata = {
            'dataset_info': self.metadata['dataset_info'] if self.metadata else {},
            'training_config': {
                'batch_load_size': self.batch_load_size,
                'validation_split': self.validation_split,
                'test_split': self.test_split,
                'incremental_training': self.incremental_training
            },
            'training_history': self.training_history,
            'model_config': {
                'spatial_shape': self.model.spatial_shape,
                'sequence_length': self.model.sequence_length,
                'embedding_dim': self.model.embedding_dim,
                'top_k_ratio': self.model.top_k_ratio
            },
            'final_timestamp': datetime.now().isoformat()
        }
        
        # Guardar en múltiples formatos
        base_name = f"large_weather_model_{timestamp}"
        
        # 1. Formato principal (Pickle)
        pickle_path = self.model_dir / f"{base_name}.pkl"
        self.model.save_model(str(pickle_path), final_metadata)
        
        # 2. SafeTensors
        try:
            safetensors_path = self.model_dir / f"{base_name}.safetensors"
            self.model.export_to_safetensors(str(safetensors_path), final_metadata)
            print(f"✅ SafeTensors guardado: {safetensors_path.name}")
        except Exception as e:
            print(f"⚠️ Error guardando SafeTensors: {e}")
        
        # 3. Historial de entrenamiento
        history_path = self.model_dir / f"{base_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(final_metadata, f, indent=2, default=str)
        
        print(f"✅ Modelo principal: {pickle_path.name}")
        print(f"✅ Historial: {history_path.name}")
        
        return str(pickle_path)
    
    def run_large_dataset_training(self):
        """Ejecuta entrenamiento completo con dataset grande."""
        print("🚀 ENTRENAMIENTO CON DATASET GRANDE")
        print("=" * 60)
        
        # Descubrir archivos de lotes
        batch_files = self.discover_batch_files()
        if not batch_files:
            print("❌ No hay datos procesados para entrenar")
            return
        
        # Crear divisiones de datos
        data_splits = self.create_data_splits(batch_files)
        
        # Crear modelo
        if not self.create_model():
            print("❌ No se pudo crear el modelo")
            return
        
        # Decidir estrategia de entrenamiento
        total_memory_needed = len(data_splits['train']) * 100 * 1024 * 1024  # Estimación
        available_memory = psutil.virtual_memory().available
        
        if total_memory_needed > available_memory * 0.7:
            print("🔄 Usando entrenamiento incremental (memoria limitada)")
            self.train_incremental(data_splits)
        else:
            print("🏋️ Usando carga completa de lotes")
            self.train_batch_loading(data_splits)
        
        # Evaluación final
        final_score = self.evaluate_final_model(data_splits)
        
        # Guardar modelo final
        model_path = self.save_final_model()
        
        # Resumen final
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"📊 Score final: R² = {final_score:.4f}")
        print(f"📁 Modelo guardado: {model_path}")
        print(f"📈 Total épocas de entrenamiento: {len(self.training_history)}")
        
        return model_path


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar modelo con dataset grande')
    
    parser.add_argument('--data-dir', default='processed_data_large',
                       help='Directorio con datos procesados')
    parser.add_argument('--model-dir', default='trained_models_large',
                       help='Directorio para modelos')
    parser.add_argument('--batch-load-size', type=int, default=5,
                       help='Lotes a cargar en memoria')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Proporción validación')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Proporción prueba')
    parser.add_argument('--incremental', action='store_true',
                       help='Forzar entrenamiento incremental')
    
    args = parser.parse_args()
    
    # Crear entrenador
    trainer = LargeDatasetTrainer(
        processed_data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_load_size=args.batch_load_size,
        validation_split=args.val_split,
        test_split=args.test_split,
        incremental_training=args.incremental
    )
    
    # Ejecutar entrenamiento
    trainer.run_large_dataset_training()


if __name__ == "__main__":
    main()