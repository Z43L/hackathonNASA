#!/usr/bin/env python3
"""
Descargador NASA IMERG - Versión Funcional
=========================================

Basado en exploración exitosa de datos disponibles.
Producto: GPM_3IMERGDF.07
Año: 2020
"""

import os
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path
import netrc
import time

class WorkingNASADownloader:
    """Descargador basado en datos que realmente funcionan."""
    
    def __init__(self, data_dir="nasa_data_working", max_workers=3):
        """Inicializar descargador."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Configuración basada en exploración exitosa
        self.base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2020"
        self.product = "GPM_3IMERGDF.07"
        self.year = "2020"
        
        # Configurar sesión con autenticación
        self.session = self._setup_session()
        
        print(f"📥 Descargador configurado:")
        print(f"   - Producto: {self.product}")
        print(f"   - Año: {self.year}")
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
    
    def get_available_days(self):
        """Obtiene días disponibles."""
        try:
            response = self.session.get(self.base_url, timeout=30)
            if response.status_code == 200:
                import re
                day_dirs = re.findall(r'href="(\d{3})/"', response.text)
                return sorted([int(d) for d in day_dirs])
            else:
                print(f"❌ Error obteniendo días: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def get_files_for_day(self, day_of_year):
        """Obtiene archivos para un día específico."""
        day_url = f"{self.base_url}/{day_of_year:03d}"
        
        try:
            response = self.session.get(day_url, timeout=15)
            if response.status_code == 200:
                import re
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
    
    def download_dataset(self, max_files=50):
        """Descarga conjunto de datos."""
        print(f"🚀 DESCARGANDO DATOS {self.product} AÑO {self.year}")
        print("=" * 50)
        
        # Obtener días disponibles
        available_days = self.get_available_days()
        
        if not available_days:
            print("❌ No se encontraron días disponibles")
            return 0
        
        print(f"📅 Días disponibles: {len(available_days)}")
        
        # Preparar descargas
        download_tasks = []
        files_count = 0
        
        for day in available_days:
            if files_count >= max_files:
                break
                
            files = self.get_files_for_day(day)
            
            for filename in files:
                if files_count >= max_files:
                    break
                    
                url = f"{self.base_url}/{day:03d}/{filename}"
                download_tasks.append((url, filename))
                files_count += 1
        
        print(f"📦 {len(download_tasks)} archivos preparados para descarga")
        
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
    
    parser = argparse.ArgumentParser(description='Descargar datos NASA IMERG funcionales')
    parser.add_argument('--max-files', type=int, default=50, help='Máximo archivos')
    parser.add_argument('--workers', type=int, default=3, help='Hilos de descarga')
    parser.add_argument('--data-dir', default='nasa_data_working', help='Directorio destino')
    
    args = parser.parse_args()
    
    downloader = WorkingNASADownloader(
        data_dir=args.data_dir,
        max_workers=args.workers
    )
    
    downloader.download_dataset(max_files=args.max_files)

if __name__ == "__main__":
    main()
