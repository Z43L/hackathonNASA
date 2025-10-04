#!/usr/bin/env python3
"""
Mecanismo de Atención Dispersa Geoespacial Avanzado
==================================================

Implementación del "Indexador Relámpago" inspirado en DeepSeek's DSA pero optimizado
para datos meteorológicos y geoespaciales. Este módulo incluye:

1. GeospatialSparseAttention: Atención dispersa consciente de la geografía
2. WeatherPatternDetector: Detector de patrones meteorológicos
3. MultiScaleAttention: Atención a múltiples escalas espaciales
4. AdaptiveSparseSelector: Selector adaptativo de regiones relevantes
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AttentionMask:
    """Estructura para almacenar máscaras de atención."""
    spatial_mask: np.ndarray  # Máscara espacial binaria
    attention_weights: np.ndarray  # Pesos de atención continuos
    selected_indices: np.ndarray  # Índices de píxeles seleccionados
    confidence_scores: np.ndarray  # Puntuaciones de confianza
    metadata: Dict  # Metadatos adicionales


class GeospatialSparseAttention:
    """
    Mecanismo de atención dispersa específicamente diseñado para datos geoespaciales.
    Incorpora conocimiento del dominio meteorológico y patrones espaciales.
    """
    
    def __init__(self,
                 spatial_shape: Tuple[int, int],
                 sparsity_ratio: float = 0.1,
                 weather_aware: bool = True,
                 multi_scale: bool = True,
                 adaptive_threshold: bool = True):
        """
        Args:
            spatial_shape: Forma espacial (height, width)
            sparsity_ratio: Proporción de píxeles a seleccionar
            weather_aware: Si usar conocimiento meteorológico
            multi_scale: Si usar atención multi-escala
            adaptive_threshold: Si usar umbralización adaptativa
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.sparsity_ratio = sparsity_ratio
        self.weather_aware = weather_aware
        self.multi_scale = multi_scale
        self.adaptive_threshold = adaptive_threshold
        
        # Calcular número de píxeles a seleccionar
        self.total_pixels = self.height * self.width
        self.k_select = max(1, int(self.total_pixels * sparsity_ratio))
        
        # Crear mapas de coordenadas
        self._create_coordinate_maps()
        
        # Inicializar detectores de patrones
        if self.weather_aware:
            self.pattern_detector = WeatherPatternDetector(spatial_shape)
        
        if self.multi_scale:
            self.multi_scale_attention = MultiScaleAttention(spatial_shape)
        
        if self.adaptive_threshold:
            self.adaptive_selector = AdaptiveSparseSelector(spatial_shape)
    
    def _create_coordinate_maps(self):
        """Crea mapas de coordenadas para operaciones espaciales."""
        # Grillas de coordenadas
        y_coords, x_coords = np.meshgrid(np.arange(self.height), 
                                       np.arange(self.width), indexing='ij')
        
        self.coord_maps = {
            'y': y_coords,
            'x': x_coords,
            'distance_center': np.sqrt((y_coords - self.height//2)**2 + 
                                     (x_coords - self.width//2)**2),
            'latitude_weight': np.cos(np.pi * y_coords / self.height),  # Peso latitudinal
        }
    
    def compute_spatial_importance(self, data_frame: np.ndarray) -> np.ndarray:
        """
        Calcula la importancia espacial basada en gradientes y variabilidad local.
        
        Args:
            data_frame: Frame de datos 2D (height, width)
            
        Returns:
            Mapa de importancia espacial
        """
        # 1. Gradientes espaciales
        grad_y, grad_x = np.gradient(data_frame)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # 2. Variabilidad local (desviación estándar en ventana deslizante)
        local_std = ndimage.generic_filter(data_frame, np.std, size=3)
        
        # 3. Curvatura (segunda derivada)
        laplacian = ndimage.laplace(data_frame)
        curvature = np.abs(laplacian)
        
        # 4. Puntos extremos locales
        local_max = ndimage.maximum_filter(data_frame, size=3) == data_frame
        local_min = ndimage.minimum_filter(data_frame, size=3) == data_frame
        extrema = (local_max | local_min).astype(float)
        
        # Combinar métricas con pesos
        importance = (
            0.3 * gradient_magnitude +
            0.3 * local_std +
            0.2 * curvature +
            0.2 * extrema
        )
        
        # Normalizar
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def compute_meteorological_relevance(self, data_frame: np.ndarray, 
                                       frame_type: str = 'precipitation') -> np.ndarray:
        """
        Calcula relevancia específica para patrones meteorológicos.
        
        Args:
            data_frame: Frame de datos 2D
            frame_type: Tipo de dato meteorológico
            
        Returns:
            Mapa de relevancia meteorológica
        """
        if not self.weather_aware:
            return np.ones_like(data_frame)
        
        return self.pattern_detector.detect_patterns(data_frame, frame_type)
    
    def compute_temporal_consistency(self, sequence: np.ndarray, 
                                   current_idx: int) -> np.ndarray:
        """
        Calcula la consistencia temporal de los píxeles.
        
        Args:
            sequence: Secuencia temporal (seq_len, height, width)
            current_idx: Índice del frame actual
            
        Returns:
            Mapa de consistencia temporal
        """
        if current_idx == 0:
            return np.ones(self.spatial_shape)
        
        # Calcular correlación temporal para cada píxel
        consistency_map = np.zeros(self.spatial_shape)
        
        # Ventana temporal para análisis
        window_start = max(0, current_idx - 5)
        window_data = sequence[window_start:current_idx+1]
        
        for i in range(self.height):
            for j in range(self.width):
                pixel_series = window_data[:, i, j]
                
                # Variabilidad temporal (menor variabilidad = mayor consistencia)
                if len(pixel_series) > 1:
                    consistency = 1.0 / (1.0 + np.std(pixel_series))
                else:
                    consistency = 1.0
                
                consistency_map[i, j] = consistency
        
        return consistency_map
    
    def lightning_indexer(self, data_frame: np.ndarray, 
                         sequence: Optional[np.ndarray] = None,
                         current_idx: int = 0,
                         frame_type: str = 'precipitation') -> AttentionMask:
        """
        Implementación del "Indexador Relámpago" optimizado para datos geoespaciales.
        
        Args:
            data_frame: Frame actual (height, width)
            sequence: Secuencia temporal completa (opcional)
            current_idx: Índice temporal actual
            frame_type: Tipo de dato meteorológico
            
        Returns:
            Máscara de atención con información completa
        """
        # 1. Importancia espacial basada en gradientes
        spatial_importance = self.compute_spatial_importance(data_frame)
        
        # 2. Relevancia meteorológica
        weather_relevance = self.compute_meteorological_relevance(data_frame, frame_type)
        
        # 3. Consistencia temporal
        if sequence is not None:
            temporal_consistency = self.compute_temporal_consistency(sequence, current_idx)
        else:
            temporal_consistency = np.ones_like(data_frame)
        
        # 4. Atención multi-escala
        if self.multi_scale:
            multiscale_attention = self.multi_scale_attention.compute_attention(data_frame)
        else:
            multiscale_attention = np.ones_like(data_frame)
        
        # Combinar todas las componentes
        combined_attention = (
            0.3 * spatial_importance +
            0.3 * weather_relevance +
            0.2 * temporal_consistency +
            0.2 * multiscale_attention
        )
        
        # Aplicar pesos geográficos (ej. mayor peso cerca del ecuador para algunos fenómenos)
        geographic_weight = self.coord_maps['latitude_weight']
        combined_attention *= geographic_weight
        
        # Selección adaptativa de píxeles
        if self.adaptive_threshold:
            selected_indices, attention_weights = self.adaptive_selector.select_pixels(
                combined_attention, self.k_select
            )
        else:
            # Selección simple por top-k
            flat_attention = combined_attention.flatten()
            top_k_indices = np.argsort(flat_attention)[-self.k_select:]
            selected_indices = np.unravel_index(top_k_indices, self.spatial_shape)
            attention_weights = flat_attention[top_k_indices]
        
        # Crear máscara binaria
        spatial_mask = np.zeros(self.spatial_shape, dtype=bool)
        if isinstance(selected_indices, tuple):
            spatial_mask[selected_indices] = True
        else:
            # Convertir índices lineales a coordenadas 2D
            coords = np.unravel_index(selected_indices, self.spatial_shape)
            spatial_mask[coords] = True
        
        # Calcular puntuaciones de confianza
        confidence_scores = self._compute_confidence_scores(
            combined_attention, selected_indices
        )
        
        # Metadatos
        metadata = {
            'sparsity_achieved': np.sum(spatial_mask) / self.total_pixels,
            'spatial_entropy': self._compute_spatial_entropy(spatial_mask),
            'selection_quality': np.mean(confidence_scores),
            'frame_type': frame_type,
            'temporal_index': current_idx
        }
        
        return AttentionMask(
            spatial_mask=spatial_mask,
            attention_weights=combined_attention,
            selected_indices=selected_indices,
            confidence_scores=confidence_scores,
            metadata=metadata
        )
    
    def _compute_confidence_scores(self, attention_map: np.ndarray, 
                                 selected_indices: Union[np.ndarray, Tuple]) -> np.ndarray:
        """Calcula puntuaciones de confianza para la selección."""
        if isinstance(selected_indices, tuple):
            selected_attention = attention_map[selected_indices]
        else:
            coords = np.unravel_index(selected_indices, self.spatial_shape)
            selected_attention = attention_map[coords]
        
        # Normalizar puntuaciones
        if len(selected_attention) > 0:
            min_val, max_val = selected_attention.min(), selected_attention.max()
            if max_val > min_val:
                confidence = (selected_attention - min_val) / (max_val - min_val)
            else:
                confidence = np.ones_like(selected_attention)
        else:
            confidence = np.array([])
        
        return confidence
    
    def _compute_spatial_entropy(self, mask: np.ndarray) -> float:
        """Calcula la entropía espacial de la selección."""
        if not np.any(mask):
            return 0.0
        
        # Dividir espacio en cuadrantes
        h_mid, w_mid = self.height // 2, self.width // 2
        
        quadrants = [
            mask[:h_mid, :w_mid],      # Top-left
            mask[:h_mid, w_mid:],      # Top-right
            mask[h_mid:, :w_mid],      # Bottom-left
            mask[h_mid:, w_mid:]       # Bottom-right
        ]
        
        # Calcular distribución
        counts = [np.sum(q) for q in quadrants]
        total = np.sum(counts)
        
        if total == 0:
            return 0.0
        
        # Calcular entropía
        probs = [c / total for c in counts if c > 0]
        entropy = -np.sum([p * np.log2(p) for p in probs])
        
        return entropy


class WeatherPatternDetector:
    """Detector de patrones meteorológicos específicos."""
    
    def __init__(self, spatial_shape: Tuple[int, int]):
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        
        # Kernels para detección de patrones
        self.kernels = self._create_pattern_kernels()
    
    def _create_pattern_kernels(self) -> Dict[str, np.ndarray]:
        """Crea kernels para detectar patrones meteorológicos."""
        kernels = {}
        
        # Kernel para frentes
        front_kernel = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ])
        kernels['front'] = front_kernel
        
        # Kernel para centros de baja presión (circular)
        low_pressure_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        kernels['low_pressure'] = low_pressure_kernel
        
        # Kernel para gradientes fuertes
        gradient_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        kernels['gradient'] = gradient_kernel
        
        return kernels
    
    def detect_patterns(self, data_frame: np.ndarray, 
                       frame_type: str = 'precipitation') -> np.ndarray:
        """
        Detecta patrones meteorológicos en el frame de datos.
        
        Args:
            data_frame: Frame de datos 2D
            frame_type: Tipo de dato ('precipitation', 'temperature', etc.)
            
        Returns:
            Mapa de relevancia basado en patrones
        """
        pattern_map = np.zeros_like(data_frame)
        
        if frame_type == 'precipitation':
            # Para precipitación, buscar frentes y gradientes
            front_response = np.abs(ndimage.convolve(data_frame, self.kernels['front']))
            gradient_response = np.abs(ndimage.convolve(data_frame, self.kernels['gradient']))
            
            # Detectar zonas de alta precipitación
            high_precip = data_frame > np.percentile(data_frame[data_frame > 0], 75)
            
            pattern_map = 0.4 * front_response + 0.4 * gradient_response + 0.2 * high_precip
            
        elif frame_type == 'pressure':
            # Para presión, buscar centros de baja/alta presión
            low_pressure_response = np.abs(ndimage.convolve(data_frame, self.kernels['low_pressure']))
            pattern_map = low_pressure_response
            
        else:
            # Patrón genérico basado en gradientes
            gradient_response = np.abs(ndimage.convolve(data_frame, self.kernels['gradient']))
            pattern_map = gradient_response
        
        # Normalizar
        if pattern_map.max() > pattern_map.min():
            pattern_map = (pattern_map - pattern_map.min()) / (pattern_map.max() - pattern_map.min())
        
        return pattern_map


class MultiScaleAttention:
    """Atención a múltiples escalas espaciales."""
    
    def __init__(self, spatial_shape: Tuple[int, int], scales: List[int] = None):
        self.spatial_shape = spatial_shape
        self.scales = scales or [1, 3, 5, 7]  # Diferentes tamaños de ventana
    
    def compute_attention(self, data_frame: np.ndarray) -> np.ndarray:
        """
        Calcula atención considerando múltiples escalas espaciales.
        
        Args:
            data_frame: Frame de datos 2D
            
        Returns:
            Mapa de atención multi-escala
        """
        multiscale_attention = np.zeros_like(data_frame)
        
        for scale in self.scales:
            if scale == 1:
                # Escala píxel individual
                scale_attention = np.abs(data_frame)
            else:
                # Promedio en ventana de tamaño scale
                scale_attention = ndimage.uniform_filter(np.abs(data_frame), size=scale)
            
            # Peso inversamente proporcional a la escala
            weight = 1.0 / scale
            multiscale_attention += weight * scale_attention
        
        # Normalizar
        multiscale_attention /= len(self.scales)
        
        return multiscale_attention


class AdaptiveSparseSelector:
    """Selector adaptativo de píxeles usando clustering."""
    
    def __init__(self, spatial_shape: Tuple[int, int], n_clusters: int = 10):
        self.spatial_shape = spatial_shape
        self.n_clusters = n_clusters
        self.height, self.width = spatial_shape
    
    def select_pixels(self, attention_map: np.ndarray, 
                     k_select: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selecciona píxeles de manera adaptativa usando clustering.
        
        Args:
            attention_map: Mapa de atención 2D
            k_select: Número de píxeles a seleccionar
            
        Returns:
            (índices_seleccionados, pesos_de_atención)
        """
        # Preparar datos para clustering
        coords = np.array(np.meshgrid(np.arange(self.height), 
                                    np.arange(self.width), indexing='ij'))
        coords_flat = coords.reshape(2, -1).T  # (N, 2)
        attention_flat = attention_map.flatten()
        
        # Filtrar píxeles con atención significativa
        significant_mask = attention_flat > np.percentile(attention_flat, 50)
        coords_filtered = coords_flat[significant_mask]
        attention_filtered = attention_flat[significant_mask]
        
        if len(coords_filtered) == 0:
            # Fallback: seleccionar top-k global
            top_indices = np.argsort(attention_flat)[-k_select:]
            return top_indices, attention_flat[top_indices]
        
        # Clustering espacial
        n_clusters = min(self.n_clusters, len(coords_filtered))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords_filtered)
            
            # Seleccionar representantes de cada cluster
            selected_indices = []
            selected_weights = []
            
            pixels_per_cluster = max(1, k_select // n_clusters)
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_coords = coords_filtered[cluster_mask]
                cluster_attention = attention_filtered[cluster_mask]
                
                if len(cluster_attention) > 0:
                    # Seleccionar los mejores píxeles del cluster
                    n_select = min(pixels_per_cluster, len(cluster_attention))
                    best_in_cluster = np.argsort(cluster_attention)[-n_select:]
                    
                    for idx in best_in_cluster:
                        coord = cluster_coords[idx]
                        linear_idx = coord[0] * self.width + coord[1]
                        selected_indices.append(linear_idx)
                        selected_weights.append(cluster_attention[idx])
            
            # Completar selección si es necesario
            remaining = k_select - len(selected_indices)
            if remaining > 0:
                all_indices = set(range(len(attention_flat)))
                used_indices = set(selected_indices)
                available_indices = list(all_indices - used_indices)
                
                if available_indices:
                    available_attention = attention_flat[available_indices]
                    additional_best = np.argsort(available_attention)[-remaining:]
                    
                    for idx in additional_best:
                        real_idx = available_indices[idx]
                        selected_indices.append(real_idx)
                        selected_weights.append(attention_flat[real_idx])
            
            return np.array(selected_indices[:k_select]), np.array(selected_weights[:k_select])
        
        else:
            # Solo un cluster: seleccionar top-k
            best_indices = np.argsort(attention_filtered)[-k_select:]
            linear_indices = []
            weights = []
            
            for idx in best_indices:
                coord = coords_filtered[idx]
                linear_idx = coord[0] * self.width + coord[1]
                linear_indices.append(linear_idx)
                weights.append(attention_filtered[idx])
            
            return np.array(linear_indices), np.array(weights)


def visualize_attention_mask(attention_mask: AttentionMask, 
                           original_data: np.ndarray,
                           save_path: Optional[str] = None):
    """
    Visualiza una máscara de atención junto con los datos originales.
    
    Args:
        attention_mask: Máscara de atención a visualizar
        original_data: Datos originales para contexto
        save_path: Ruta para guardar la visualización (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Datos originales
    im1 = axes[0, 0].imshow(original_data, cmap='viridis')
    axes[0, 0].set_title('Datos Originales')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Mapa de atención
    im2 = axes[0, 1].imshow(attention_mask.attention_weights, cmap='hot')
    axes[0, 1].set_title('Mapa de Atención')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Máscara binaria
    axes[1, 0].imshow(attention_mask.spatial_mask, cmap='binary')
    axes[1, 0].set_title(f'Píxeles Seleccionados ({np.sum(attention_mask.spatial_mask)})')
    
    # Datos con máscara superpuesta
    masked_data = original_data.copy()
    masked_data[~attention_mask.spatial_mask] *= 0.3  # Atenuar píxeles no seleccionados
    im4 = axes[1, 1].imshow(masked_data, cmap='viridis')
    axes[1, 1].set_title('Datos con Atención Aplicada')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Información adicional
    metadata = attention_mask.metadata
    info_text = f"""Sparsity: {metadata['sparsity_achieved']:.3f}
Entropy: {metadata['spatial_entropy']:.3f}
Quality: {metadata['selection_quality']:.3f}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en {save_path}")
    
    plt.show()


def test_sparse_attention():
    """Función de prueba para el mecanismo de atención dispersa."""
    
    print("=== PROBANDO ATENCIÓN DISPERSA GEOESPACIAL ===")
    
    # Crear datos sintéticos que simulan patrones meteorológicos
    height, width = 50, 50
    np.random.seed(42)
    
    # Simular campo de precipitación con frentes
    x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
    
    # Frente principal (diagonal)
    front = np.exp(-((x - y - 2)**2) / 2) * 10
    
    # Células convectivas (puntos de alta intensidad)
    cells = (np.exp(-((x - 3)**2 + (y - 7)**2) / 0.5) * 15 +
             np.exp(-((x - 7)**2 + (y - 3)**2) / 0.8) * 12)
    
    # Ruido de fondo
    noise = np.random.randn(height, width) * 0.5
    
    # Combinar patrones
    precipitation_data = front + cells + noise
    precipitation_data = np.maximum(precipitation_data, 0)  # No precipitación negativa
    
    print(f"Datos sintéticos creados: {precipitation_data.shape}")
    print(f"Precipitación - Min: {precipitation_data.min():.2f}, Max: {precipitation_data.max():.2f}")
    
    # Crear instancia de atención dispersa
    sparse_attention = GeospatialSparseAttention(
        spatial_shape=(height, width),
        sparsity_ratio=0.15,  # Seleccionar 15% de píxeles
        weather_aware=True,
        multi_scale=True,
        adaptive_threshold=True
    )
    
    # Aplicar atención dispersa
    attention_mask = sparse_attention.lightning_indexer(
        precipitation_data, 
        frame_type='precipitation'
    )
    
    print(f"\nResultados de atención dispersa:")
    print(f"Píxeles seleccionados: {np.sum(attention_mask.spatial_mask)} de {height * width}")
    print(f"Sparsity lograda: {attention_mask.metadata['sparsity_achieved']:.3f}")
    print(f"Entropía espacial: {attention_mask.metadata['spatial_entropy']:.3f}")
    print(f"Calidad de selección: {attention_mask.metadata['selection_quality']:.3f}")
    
    # Visualizar resultados
    visualize_attention_mask(attention_mask, precipitation_data)
    
    # Probar con secuencia temporal
    print("\n=== PROBANDO SECUENCIA TEMPORAL ===")
    
    sequence_length = 10
    temporal_sequence = np.zeros((sequence_length, height, width))
    
    for t in range(sequence_length):
        # Simular movimiento del frente
        offset = t * 0.5
        moving_front = np.exp(-((x - y - 2 + offset)**2) / 2) * 10
        temporal_sequence[t] = moving_front + cells + np.random.randn(height, width) * 0.3
        temporal_sequence[t] = np.maximum(temporal_sequence[t], 0)
    
    # Aplicar atención con contexto temporal
    final_mask = sparse_attention.lightning_indexer(
        temporal_sequence[-1],  # Frame actual
        sequence=temporal_sequence,
        current_idx=sequence_length - 1,
        frame_type='precipitation'
    )
    
    print(f"Atención con contexto temporal:")
    print(f"Píxeles seleccionados: {np.sum(final_mask.spatial_mask)}")
    print(f"Sparsity: {final_mask.metadata['sparsity_achieved']:.3f}")
    
    # Comparar selección con y sin contexto temporal
    spatial_only_mask = sparse_attention.lightning_indexer(
        temporal_sequence[-1],
        frame_type='precipitation'
    )
    
    overlap = np.sum(final_mask.spatial_mask & spatial_only_mask.spatial_mask)
    total_selected = np.sum(final_mask.spatial_mask)
    
    print(f"Overlap entre selección temporal y espacial: {overlap}/{total_selected} ({overlap/total_selected:.2%})")
    
    print("\n=== PRUEBAS COMPLETADAS ===")


if __name__ == "__main__":
    test_sparse_attention()