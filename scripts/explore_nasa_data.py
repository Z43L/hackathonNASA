#!/usr/bin/env python3
"""
Explorador de Datos NASA IMERG
=============================

Script para explorar qué datos están disponibles en el servidor NASA
y encontrar las URLs correctas para descarga.
"""

import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys

class NASADataExplorer:
    """Explorador de datos NASA IMERG."""
    
    def __init__(self):
        """Inicializar explorador."""
        self.session = requests.Session()
        
        # URLs base conocidas
        self.base_urls = {
            'daily_v06': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06',
            'daily_v07': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07',
            'monthly_v06': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.06',
            'monthly_v07': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07'
        }
        
        # Verificar credenciales
        netrc_file = Path.home() / '.netrc'
        if not netrc_file.exists():
            print("❌ Credenciales NASA no encontradas. Ejecuta setup_credentials.py primero")
            sys.exit(1)
    
    def check_url_exists(self, url):
        """Verifica si una URL existe."""
        try:
            response = self.session.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def explore_directory_structure(self, base_url, year=2023):
        """Explora la estructura de directorios para un año."""
        print(f"🔍 Explorando estructura para {year} en {base_url}")
        
        year_url = f"{base_url}/{year}"
        
        try:
            response = self.session.get(year_url, timeout=30)
            if response.status_code == 200:
                # Buscar enlaces a subdirectorios
                content = response.text
                
                # Buscar patrones de directorios (números de 3 dígitos para días del año)
                day_dirs = re.findall(r'href="(\d{3})/"', content)
                month_dirs = re.findall(r'href="(\d{2})/"', content)
                
                if day_dirs:
                    print(f"  📅 Estructura por día del año: {len(day_dirs)} directorios")
                    print(f"  📂 Primeros directorios: {day_dirs[:5]}")
                    return 'daily', day_dirs
                elif month_dirs:
                    print(f"  📅 Estructura por mes: {len(month_dirs)} directorios")
                    print(f"  📂 Directorios de mes: {month_dirs}")
                    return 'monthly', month_dirs
                else:
                    print("  ❓ Estructura no reconocida")
                    return None, []
            else:
                print(f"  ❌ Error accediendo a {year_url}: {response.status_code}")
                return None, []
        except Exception as e:
            print(f"  ❌ Error explorando {year_url}: {e}")
            return None, []
    
    def find_available_files(self, base_url, year=2023, month=None, day_of_year=None):
        """Encuentra archivos disponibles en un directorio específico."""
        if month:
            dir_url = f"{base_url}/{year}/{month:02d}"
        elif day_of_year:
            dir_url = f"{base_url}/{year}/{day_of_year:03d}"
        else:
            dir_url = f"{base_url}/{year}"
        
        print(f"📂 Buscando archivos en {dir_url}")
        
        try:
            response = self.session.get(dir_url, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                # Buscar archivos HDF5
                hdf5_files = re.findall(r'href="([^"]*\.HDF5)"', content)
                
                if hdf5_files:
                    print(f"  ✅ Encontrados {len(hdf5_files)} archivos HDF5")
                    print(f"  📄 Ejemplos: {hdf5_files[:3]}")
                    return hdf5_files
                else:
                    print("  📭 No se encontraron archivos HDF5")
                    return []
            else:
                print(f"  ❌ Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return []
    
    def test_sample_downloads(self, base_url, year=2023):
        """Prueba descargar algunos archivos de muestra."""
        print(f"🧪 Probando descargas de muestra para {year}")
        
        # Explorar estructura
        structure_type, dirs = self.explore_directory_structure(base_url, year)
        
        if not dirs:
            print("  ❌ No se pudo determinar estructura de directorios")
            return []
        
        successful_urls = []
        
        # Probar primeros directorios
        for dir_name in dirs[:5]:
            if structure_type == 'daily':
                day_of_year = int(dir_name)
                files = self.find_available_files(base_url, year, day_of_year=day_of_year)
            else:
                month = int(dir_name)
                files = self.find_available_files(base_url, year, month=month)
            
            if files:
                # Probar descargar primer archivo (solo HEAD request)
                first_file = files[0]
                if structure_type == 'daily':
                    test_url = f"{base_url}/{year}/{dir_name}/{first_file}"
                else:
                    test_url = f"{base_url}/{year}/{dir_name}/{first_file}"
                
                if self.check_url_exists(test_url):
                    print(f"  ✅ URL válida: {test_url}")
                    successful_urls.append(test_url)
                else:
                    print(f"  ❌ URL inválida: {test_url}")
        
        return successful_urls
    
    def explore_all_versions(self, year=2023):
        """Explora todas las versiones de datos disponibles."""
        print("🌍 EXPLORANDO TODAS LAS VERSIONES DE DATOS NASA IMERG")
        print("=" * 60)
        
        results = {}
        
        for version_name, base_url in self.base_urls.items():
            print(f"\n--- {version_name.upper()} ---")
            
            successful_urls = self.test_sample_downloads(base_url, year)
            results[version_name] = successful_urls
            
            if successful_urls:
                print(f"  🎉 {len(successful_urls)} URLs exitosas encontradas")
            else:
                print("  💔 No se encontraron URLs válidas")
        
        # Resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE EXPLORACIÓN")
        print("=" * 60)
        
        working_versions = []
        for version, urls in results.items():
            if urls:
                working_versions.append(version)
                print(f"✅ {version}: {len(urls)} URLs funcionando")
            else:
                print(f"❌ {version}: Sin URLs válidas")
        
        if working_versions:
            print(f"\n🎯 RECOMENDACIÓN:")
            best_version = working_versions[0]
            print(f"Usar versión: {best_version}")
            print(f"URL base: {self.base_urls[best_version]}")
            
            # Mostrar ejemplos de URLs exitosas
            print(f"\n📄 URLs de ejemplo:")
            for url in results[best_version][:3]:
                print(f"  {url}")
            
            return best_version, self.base_urls[best_version]
        else:
            print("\n💥 No se encontraron versiones funcionales")
            return None, None
    
    def generate_working_urls(self, base_url, year=2023, start_month=1, end_month=12):
        """Genera lista de URLs que funcionan para un período."""
        print(f"\n🔧 GENERANDO URLs PARA DESCARGA")
        print("=" * 40)
        
        working_urls = []
        
        # Determinar estructura
        structure_type, dirs = self.explore_directory_structure(base_url, year)
        
        if structure_type == 'daily':
            # Generar para días del año
            for doy in range(1, 366):  # Días del año
                if f"{doy:03d}" in dirs:
                    files = self.find_available_files(base_url, year, day_of_year=doy)
                    for file in files:
                        url = f"{base_url}/{year}/{doy:03d}/{file}"
                        working_urls.append(url)
        
        elif structure_type == 'monthly':
            # Generar para meses
            for month in range(start_month, end_month + 1):
                if f"{month:02d}" in dirs:
                    files = self.find_available_files(base_url, year, month=month)
                    for file in files:
                        url = f"{base_url}/{year}/{month:02d}/{file}"
                        working_urls.append(url)
        
        print(f"📋 Generadas {len(working_urls)} URLs de descarga")
        
        # Guardar en archivo
        urls_file = Path("nasa_urls_working.txt")
        with open(urls_file, 'w') as f:
            for url in working_urls:
                f.write(url + '\n')
        
        print(f"💾 URLs guardadas en: {urls_file}")
        
        return working_urls


def main():
    """Función principal."""
    explorer = NASADataExplorer()
    
    # Explorar todas las versiones
    best_version, best_url = explorer.explore_all_versions(2023)
    
    if best_version and best_url:
        # Generar URLs de trabajo
        working_urls = explorer.generate_working_urls(best_url, 2023)
        
        print(f"\n🚀 SIGUIENTE PASO:")
        print(f"Modifica el script download_large_dataset.py para usar:")
        print(f"  Base URL: {best_url}")
        print(f"  Versión: {best_version}")
        print(f"  URLs generadas: {len(working_urls)}")
    else:
        print("\n💡 ALTERNATIVAS:")
        print("1. Verificar credenciales NASA Earthdata")
        print("2. Probar con años más recientes (2024)")
        print("3. Usar datos mensuales en lugar de diarios")


if __name__ == "__main__":
    main()