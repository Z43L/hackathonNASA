#!/usr/bin/env python3
"""
Descargador NASA IMERG Mensual - FUNCIONAL
==========================================

Descargador para datos NASA IMERG mensuales que realmente funcionan.
Producto: GPM_3IMERGM.07 (datos mensuales con archivos directos)
"""

import os
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path
import netrc
import time
import re

class FunctionalNASADownloader:
    """Descargador para datos NASA IMERG mensuales funcionales."""
    
    def __init__(self, data_dir="nasa_data_functional", max_workers=3):
        """Inicializar descargador."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Configuración para producto mensual funcional
        self.base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07"
        self.product = "GPM_3IMERGM.07"
        self.available_years = [2020, 2021, 2022, 2023, 2024]
        
        # Configurar sesión con autenticación
        self.session = self._setup_session()
        
        print(f"📥 Descargador funcional configurado:")
        print(f"   - Producto: {self.product} (Mensual)")
        print(f"   - Años disponibles: {self.available_years}")
        print(f"   - URL base: {self.base_url}")
        print(f"   - Directorio: {self.data_dir}")
    
    def _setup_session(self):
        """Configura sesión con credenciales."""
        session = requests.Session()
        
        try:
            netrc_path = Path.home() / '.netrc'
            if netrc_path.exists():
                auth_info = netrc.netrc()
                username, _, password = auth_info.authenticators('urs.earthdata.nasa.gov')
                session.auth = (username, password)
                print("🔑 Credenciales NASA configuradas desde .netrc")
            else:
                print("⚠️ No se encontraron credenciales .netrc")
        except Exception as e:
            print(f"⚠️ Error configurando credenciales: {e}")
        
        return session
    
    def get_files_for_year(self, year):
        """Obtiene archivos HDF5 directamente para un año."""
        year_url = f"{self.base_url}/{year}"
        
        try:
            print(f"📅 Explorando {year}...")
            response = self.session.get(year_url, timeout=30)
            
            if response.status_code == 200:
                # Buscar archivos HDF5 directamente
                files = re.findall(r'href="([^"]*\.HDF5)"', response.text)
                
                # Filtrar duplicados si existen
                unique_files = list(set(files))
                
                print(f"✅ {year}: {len(unique_files)} archivos encontrados")
                return unique_files
            else:
                print(f"❌ Error {year}: status {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error explorando {year}: {e}")
            return []
    
    def download_file(self, url, filename):
        """Descarga un archivo HDF5."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"⏭️ {filename} ya existe ({size_mb:.1f} MB)")
            return True
        
        try:
            print(f"📥 Descargando {filename}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Descargar con progreso
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} completado ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_dataset(self, max_files=100, years=None):
        """Descarga conjunto de datos mensuales."""
        print(f"🚀 DESCARGANDO DATOS NASA IMERG MENSUALES")
        print("=" * 50)
        
        if years is None:
            years = self.available_years
        
        # Recopilar todos los archivos disponibles
        all_files = []
        
        for year in years:
            files = self.get_files_for_year(year)
            
            for filename in files:
                if len(all_files) >= max_files:
                    break
                    
                url = f"{self.base_url}/{year}/{filename}"
                all_files.append((url, filename))
        
        if not all_files:
            print("❌ No se encontraron archivos para descargar")
            return 0
        
        print(f"\n📦 {len(all_files)} archivos preparados para descarga")
        print(f"🎯 Descargando primeros {min(len(all_files), max_files)} archivos")
        
        # Ejecutar descargas en paralelo
        successful = 0
        failed = 0
        
        download_tasks = all_files[:max_files]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.download_file, url, filename): filename
                for url, filename in download_tasks
            }
            
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
        
        # Resumen de descarga
        print(f"\n📊 DESCARGA COMPLETADA:")
        print(f"✅ Exitosas: {successful}")
        print(f"❌ Fallidas: {failed}")
        print(f"📁 Directorio: {self.data_dir.absolute()}")
        
        # Verificar archivos descargados
        downloaded_files = list(self.data_dir.glob("*.HDF5"))
        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024 * 1024 * 1024)
        
        print(f"📄 Archivos totales: {len(downloaded_files)}")
        print(f"💾 Tamaño total: {total_size:.2f} GB")
        
        if successful > 0:
            print(f"\n🎉 ¡{successful} archivos NASA descargados exitosamente!")
            print(f"\n🚀 SIGUIENTE PASO:")
            print(f"   python scripts/preprocess_large_dataset.py --raw-dir {self.data_dir}")
            print(f"\n💡 ARCHIVOS DISPONIBLES:")
            for f in downloaded_files[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.1f} MB)")
            
            if len(downloaded_files) > 5:
                print(f"   ... y {len(downloaded_files) - 5} archivos más")
        
        return successful

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Descargar datos NASA IMERG mensuales funcionales')
    parser.add_argument('--max-files', type=int, default=100, help='Máximo archivos a descargar')
    parser.add_argument('--workers', type=int, default=3, help='Hilos de descarga paralelos')
    parser.add_argument('--data-dir', default='nasa_data_functional', help='Directorio destino')
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021, 2022], 
                       help='Años a descargar (ej: --years 2022 2023)')
    
    args = parser.parse_args()
    
    print(f"🌍 NASA IMERG DOWNLOADER - VERSIÓN FUNCIONAL")
    print(f"=" * 50)
    print(f"📊 Configuración:")
    print(f"   - Máximo archivos: {args.max_files}")
    print(f"   - Hilos paralelos: {args.workers}")
    print(f"   - Años objetivo: {args.years}")
    print(f"   - Directorio: {args.data_dir}")
    print(f"")
    
    downloader = FunctionalNASADownloader(
        data_dir=args.data_dir,
        max_workers=args.workers
    )
    
    success_count = downloader.download_dataset(
        max_files=args.max_files, 
        years=args.years
    )
    
    if success_count > 0:
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"✅ Descarga exitosa: {success_count} archivos")
        print(f"🗂️ Datos listos para procesamiento")
        print(f"🚀 Continúa con el preprocesamiento para entrenar tu modelo")
    else:
        print(f"\n💥 DESCARGA FALLIDA")
        print(f"💡 Verifica credenciales NASA Earthdata")

if __name__ == "__main__":
    main()