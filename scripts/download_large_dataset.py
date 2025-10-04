#!/usr/bin/env python3
"""
Descarga de Dataset Grande NASA IMERG
====================================

Script para descargar grandes volúmenes de datos IMERG para entrenar
modelos de predicción meteorológica con más datos.

Configuraciones disponibles:
- Pequeño: 1 mes (30 archivos, ~500MB)
- Mediano: 6 meses (180 archivos, ~3GB)  
- Grande: 1 año (365 archivos, ~6GB)
- Muy Grande: 2 años (730 archivos, ~12GB)
"""

import os
import sys
import requests
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
import time
import argparse

class LargeDatasetDownloader:
    """Descargador de datasets grandes NASA IMERG."""
    
    def __init__(self, data_dir="raw_data_large", max_workers=4):
        """
        Inicializar descargador.
        
        Args:
            data_dir: Directorio para datos
            max_workers: Hilos de descarga paralela
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # URLs base para diferentes productos IMERG
        self.base_urls = {
            'daily': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06',
            'monthly': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.06'
        }
        
        self.session = requests.Session()
        
        # Verificar credenciales
        netrc_file = Path.home() / '.netrc'
        if not netrc_file.exists():
            print("❌ Credenciales NASA no encontradas. Ejecuta setup_credentials.py primero")
            sys.exit(1)
    
    def generate_date_range(self, start_date, end_date, product='daily'):
        """
        Genera rango de fechas para descargar.
        
        Args:
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            product: 'daily' o 'monthly'
            
        Returns:
            Lista de fechas
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        
        while current <= end:
            dates.append(current)
            if product == 'daily':
                current += timedelta(days=1)
            else:  # monthly
                # Próximo mes
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        
        return dates
    
    def build_download_url(self, date, product='daily'):
        """
        Construye URL de descarga para fecha específica.
        
        Args:
            date: Objeto datetime
            product: 'daily' o 'monthly'
            
        Returns:
            URL completa y nombre de archivo
        """
        year = date.year
        month = date.month
        day = date.day
        
        if product == 'daily':
            # Día del año (DOY) para estructura de directorios
            day_of_year = date.timetuple().tm_yday
            
            # Formato diario correcto: 3B-DAY.MS.MRG.3IMERG.20231101-S000000-E235959.V06B.HDF5
            filename = f"3B-DAY.MS.MRG.3IMERG.{year:04d}{month:02d}{day:02d}-S000000-E235959.V06B.HDF5"
            url = f"{self.base_urls['daily']}/{year}/{day_of_year:03d}/{filename}"
        else:
            # Formato mensual: 3B-MO.MS.MRG.3IMERG.20231101-S000000-E235959.11.V06B.HDF5
            filename = f"3B-MO.MS.MRG.3IMERG.{year:04d}{month:02d}01-S000000-E235959.{month:02d}.V06B.HDF5"
            url = f"{self.base_urls['monthly']}/{year}/{filename}"
        
        return url, filename
    
    def download_file(self, url, filename, max_retries=3):
        """
        Descarga un archivo con reintentos.
        
        Args:
            url: URL del archivo
            filename: Nombre del archivo local
            max_retries: Máximo número de reintentos
            
        Returns:
            True si exitoso, False si falló
        """
        filepath = self.data_dir / filename
        
        # Verificar si ya existe
        if filepath.exists():
            print(f"⏭️  Saltando {filename} (ya existe)")
            return True
        
        for attempt in range(max_retries):
            try:
                print(f"📥 Descargando {filename} (intento {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Obtener tamaño total
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progreso cada 10MB
                            if downloaded % (10 * 1024 * 1024) == 0:
                                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                                print(f"  📊 {progress:.1f}% - {downloaded // (1024*1024)} MB")
                
                print(f"✅ {filename} descargado ({downloaded // (1024*1024)} MB)")
                return True
                
            except Exception as e:
                print(f"❌ Error descargando {filename}: {e}")
                if filepath.exists():
                    filepath.unlink()  # Eliminar archivo parcial
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Backoff exponencial
                    print(f"⏳ Esperando {wait_time}s antes de reintentar...")
                    time.sleep(wait_time)
        
        print(f"💥 Falló descarga de {filename} después de {max_retries} intentos")
        return False
    
    def download_parallel(self, urls_and_files):
        """
        Descarga archivos en paralelo.
        
        Args:
            urls_and_files: Lista de tuplas (url, filename)
            
        Returns:
            Estadísticas de descarga
        """
        successful = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar tareas
            future_to_file = {
                executor.submit(self.download_file, url, filename): filename
                for url, filename in urls_and_files
            }
            
            # Procesar resultados
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"❌ Excepción en {filename}: {e}")
                    failed += 1
        
        return successful, failed
    
    def download_dataset(self, dataset_size='medium', start_date=None, end_date=None, product='daily'):
        """
        Descarga dataset según configuración.
        
        Args:
            dataset_size: 'small', 'medium', 'large', 'xlarge' o 'custom'
            start_date: Fecha inicio para 'custom' (YYYY-MM-DD)
            end_date: Fecha fin para 'custom' (YYYY-MM-DD)
            product: 'daily' o 'monthly'
        """
        print(f"🚀 INICIANDO DESCARGA DE DATASET {dataset_size.upper()}")
        print("=" * 60)
        
        # Configuraciones predefinidas
        configs = {
            'small': ('2023-10-01', '2023-10-30'),    # 1 mes
            'medium': ('2023-05-01', '2023-10-31'),   # 6 meses  
            'large': ('2023-01-01', '2023-12-31'),    # 1 año
            'xlarge': ('2022-01-01', '2023-12-31'),   # 2 años
        }
        
        if dataset_size == 'custom':
            if not start_date or not end_date:
                print("❌ Para dataset custom, especifica start_date y end_date")
                return
            dates_config = (start_date, end_date)
        else:
            dates_config = configs.get(dataset_size)
            if not dates_config:
                print(f"❌ Tamaño de dataset no válido: {dataset_size}")
                print(f"Opciones: {list(configs.keys()) + ['custom']}")
                return
        
        start_date, end_date = dates_config
        
        # Generar fechas
        dates = self.generate_date_range(start_date, end_date, product)
        
        print(f"📅 Período: {start_date} a {end_date}")
        print(f"📊 Producto: {product}")
        print(f"🗂️  Total archivos: {len(dates)}")
        
        # Estimar tamaño
        avg_size_mb = 15 if product == 'daily' else 450  # MB promedio por archivo
        estimated_size_gb = (len(dates) * avg_size_mb) / 1024
        print(f"💾 Tamaño estimado: {estimated_size_gb:.1f} GB")
        
        # Confirmar descarga
        response = input(f"\n¿Continuar con la descarga? (y/N): ")
        if response.lower() != 'y':
            print("❌ Descarga cancelada")
            return
        
        # Construir URLs
        urls_and_files = []
        for date in dates:
            url, filename = self.build_download_url(date, product)
            urls_and_files.append((url, filename))
        
        # Iniciar descarga
        start_time = time.time()
        successful, failed = self.download_parallel(urls_and_files)
        total_time = time.time() - start_time
        
        # Estadísticas finales
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS DE DESCARGA")
        print("=" * 60)
        print(f"✅ Archivos descargados: {successful}")
        print(f"❌ Archivos fallidos: {failed}")
        print(f"⏱️  Tiempo total: {total_time / 60:.1f} minutos")
        print(f"📁 Directorio: {self.data_dir.absolute()}")
        
        # Verificar espacio en disco
        total_size = sum(f.stat().st_size for f in self.data_dir.glob('*.HDF5'))
        print(f"💾 Espacio usado: {total_size / (1024**3):.2f} GB")
        
        if successful > 0:
            print(f"\n🎉 ¡Descarga completada! Listos para entrenar con {successful} archivos")
            print("\n🚀 PRÓXIMOS PASOS:")
            print("1. python scripts/data_preprocessing.py")
            print("2. python scripts/train_model.py")
        else:
            print("\n💥 No se descargó ningún archivo exitosamente")


def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Descargar dataset grande NASA IMERG')
    
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'xlarge', 'custom'],
                       default='medium', help='Tamaño del dataset')
    parser.add_argument('--start', help='Fecha inicio para custom (YYYY-MM-DD)')
    parser.add_argument('--end', help='Fecha fin para custom (YYYY-MM-DD)')
    parser.add_argument('--product', choices=['daily', 'monthly'], default='daily',
                       help='Tipo de producto IMERG')
    parser.add_argument('--workers', type=int, default=4,
                       help='Número de descargas paralelas')
    parser.add_argument('--data-dir', default='raw_data_large',
                       help='Directorio para datos')
    
    args = parser.parse_args()
    
    # Crear descargador
    downloader = LargeDatasetDownloader(
        data_dir=args.data_dir,
        max_workers=args.workers
    )
    
    # Descargar dataset
    downloader.download_dataset(
        dataset_size=args.size,
        start_date=args.start,
        end_date=args.end,
        product=args.product
    )


if __name__ == "__main__":
    main()