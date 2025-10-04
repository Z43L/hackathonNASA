#!/usr/bin/env python3
"""
Descargador NASA IMERG - Versión Mensual Funcional
==================================================

Descargador corregido para datos IMERG con estructura mensual.
"""

import os
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path
import netrc
import time
import re

class MonthlyNASADownloader:
    """Descargador para datos NASA IMERG mensuales."""
    
    def __init__(self, data_dir="nasa_data_monthly", max_workers=3):
        """Inicializar descargador."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Configuración basada en exploración exitosa
        self.base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
        self.product = "GPM_3IMERGDF.07"
        self.available_years = [2020, 2021, 2022]  # Años que sabemos que funcionan
        
        # Configurar sesión con autenticación
        self.session = self._setup_session()
        
        print(f"📥 Descargador mensual configurado:")
        print(f"   - Producto: {self.product}")
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
                print("🔑 Credenciales configuradas desde .netrc")
            else:
                print("⚠️ No se encontraron credenciales .netrc")
        except Exception as e:
            print(f"⚠️ Error configurando credenciales: {e}")
        
        return session
    
    def get_available_months(self, year):
        """Obtiene meses disponibles para un año."""
        year_url = f"{self.base_url}/{year}"
        
        try:
            response = self.session.get(year_url, timeout=30)
            if response.status_code == 200:
                # Buscar directorios de meses (formato 01, 02, etc.)
                month_dirs = re.findall(r'href="(\d{2})/"', response.text)
                return sorted([int(m) for m in month_dirs])
            else:
                print(f"❌ Error obteniendo meses para {year}: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def get_files_for_month(self, year, month):
        """Obtiene archivos para un mes específico."""
        month_url = f"{self.base_url}/{year}/{month:02d}"
        
        try:
            response = self.session.get(month_url, timeout=15)
            if response.status_code == 200:
                # Buscar archivos HDF5
                files = re.findall(r'href="([^"]*\.HDF5)"', response.text)
                return files
            else:
                return []
        except Exception as e:
            return []
    
    def download_file(self, url, filename):
        """Descarga un archivo."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"⏭️ {filename} ya existe")
            return True
        
        try:
            print(f"📥 Descargando {filename}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} completado ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_dataset(self, max_files=50, years=None):
        """Descarga conjunto de datos mensuales."""
        print(f"🚀 DESCARGANDO DATOS {self.product}")
        print("=" * 50)
        
        if years is None:
            years = self.available_years
        
        # Preparar descargas
        download_tasks = []
        files_count = 0
        
        for year in years:
            if files_count >= max_files:
                break
                
            print(f"\n📅 Explorando año {year}...")
            months = self.get_available_months(year)
            
            if not months:
                print(f"❌ No se encontraron meses para {year}")
                continue
                
            print(f"📊 Meses disponibles: {len(months)} ({min(months)}-{max(months)})")
            
            for month in months:
                if files_count >= max_files:
                    break
                    
                files = self.get_files_for_month(year, month)
                
                if not files:
                    continue
                    
                print(f"   📁 {year}-{month:02d}: {len(files)} archivos")
                
                for filename in files:
                    if files_count >= max_files:
                        break
                        
                    url = f"{self.base_url}/{year}/{month:02d}/{filename}"
                    download_tasks.append((url, filename))
                    files_count += 1
        
        if not download_tasks:
            print("❌ No se encontraron archivos para descargar")
            return 0
        
        print(f"\n📦 {len(download_tasks)} archivos preparados para descarga")
        
        # Ejecutar descargas
        successful = 0
        failed = 0
        
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
        
        # Resumen
        print(f"\n📊 DESCARGA COMPLETADA:")
        print(f"✅ Exitosas: {successful}")
        print(f"❌ Fallidas: {failed}")
        print(f"📁 Directorio: {self.data_dir.absolute()}")
        
        if successful > 0:
            print(f"\n🎉 ¡{successful} archivos descargados!")
            print("🚀 SIGUIENTE PASO:")
            print(f"   python scripts/preprocess_large_dataset.py --raw-dir {self.data_dir}")
        
        return successful

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Descargar datos NASA IMERG mensuales')
    parser.add_argument('--max-files', type=int, default=50, help='Máximo archivos')
    parser.add_argument('--workers', type=int, default=3, help='Hilos de descarga')
    parser.add_argument('--data-dir', default='nasa_data_monthly', help='Directorio destino')
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021, 2022], 
                       help='Años a descargar')
    
    args = parser.parse_args()
    
    downloader = MonthlyNASADownloader(
        data_dir=args.data_dir,
        max_workers=args.workers
    )
    
    downloader.download_dataset(max_files=args.max_files, years=args.years)

if __name__ == "__main__":
    main()