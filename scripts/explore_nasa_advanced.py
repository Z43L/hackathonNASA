#!/usr/bin/env python3
"""
Explorador Avanzado de Datos NASA IMERG
======================================

Script mejorado para encontrar qué años y datos están realmente disponibles.
"""

import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys
import netrc
import os

def setup_nasa_session():
    """Configura sesión con credenciales NASA."""
    session = requests.Session()
    
    try:
        # Intentar leer credenciales desde .netrc
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            print("🔑 Usando credenciales desde .netrc")
            auth_info = netrc.netrc()
            username, _, password = auth_info.authenticators('urs.earthdata.nasa.gov')
            session.auth = (username, password)
            return session, True
        else:
            print("⚠️ No se encontraron credenciales en .netrc")
            return session, False
    except Exception as e:
        print(f"⚠️ Error configurando credenciales: {e}")
        return session, False

def explore_alternative_sources():
    """Explora fuentes alternativas de datos NASA."""
    print("\n🔍 EXPLORANDO FUENTES ALTERNATIVAS")
    print("=" * 40)
    
    session, has_auth = setup_nasa_session()
    
    alternative_urls = [
        # Diferentes servidores NASA
        'https://disc2.gesdisc.eosdis.nasa.gov/data/GPM_L3',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3',
        'https://giovanni.gsfc.nasa.gov/giovanni',
        
        # Diferentes productos IMERG
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06',  # Half-hourly
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07',  # Half-hourly V07
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07',  # Daily V07
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07',   # Monthly V07
    ]
    
    working_urls = []
    
    for url in alternative_urls:
        print(f"\nProbando: {url}")
        try:
            response = session.get(url, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Accesible - Status: {response.status_code}")
                
                # Buscar estructura de directorios
                content = response.text
                years = re.findall(r'href="(\d{4})/"', content)
                
                if years:
                    years = sorted([int(y) for y in years])
                    print(f"📅 Años encontrados: {years[0]}-{years[-1]} ({len(years)} años)")
                    working_urls.append((url, years))
                else:
                    print("📁 Accesible pero sin estructura de años clara")
                    
            elif response.status_code == 401:
                print(f"🔐 Requiere autenticación - Status: {response.status_code}")
                
            elif response.status_code == 403:
                print(f"🚫 Acceso denegado - Status: {response.status_code}")
                
            else:
                print(f"❌ Error - Status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return working_urls

def test_specific_data_access():
    """Prueba acceso a datos específicos."""
    print("\n🎯 PROBANDO ACCESO A DATOS ESPECÍFICOS")
    print("=" * 40)
    
    session, has_auth = setup_nasa_session()
    
    # URLs específicas conocidas para probar
    test_urls = [
        # Datos más antiguos que podrían estar disponibles
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2014',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2015',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2020',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2021',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/2022',
        
        # Versión 07
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2020',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2021',
        'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2022',
    ]
    
    working_data = []
    
    for url in test_urls:
        year = url.split('/')[-1]
        product = url.split('/')[-2]
        
        print(f"\nProbando {product} año {year}...")
        
        try:
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                content = response.text
                
                # Buscar días/directorios
                day_dirs = re.findall(r'href="(\d{3})/"', content)
                month_dirs = re.findall(r'href="(\d{2})/"', content)
                
                if day_dirs:
                    print(f"✅ Datos diarios disponibles: {len(day_dirs)} días")
                    
                    # Probar acceso a archivos específicos
                    test_day = day_dirs[0]
                    day_url = f"{url}/{test_day}"
                    
                    day_response = session.get(day_url, timeout=10)
                    if day_response.status_code == 200:
                        files = re.findall(r'href="([^"]*\.HDF5)"', day_response.text)
                        if files:
                            print(f"📄 Archivos HDF5: {len(files)}")
                            example_file_url = f"{day_url}/{files[0]}"
                            working_data.append({
                                'year': year,
                                'product': product,
                                'base_url': url,
                                'example_file': example_file_url,
                                'total_days': len(day_dirs)
                            })
                            print(f"🔗 URL de ejemplo: {example_file_url}")
                        
                elif month_dirs:
                    print(f"✅ Estructura mensual: {len(month_dirs)} meses")
                    working_data.append({
                        'year': year,
                        'product': product,
                        'base_url': url,
                        'type': 'monthly',
                        'total_months': len(month_dirs)
                    })
                    
                else:
                    print("❓ Estructura no reconocida")
                    
            else:
                print(f"❌ Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return working_data

def create_working_downloader(working_data):
    """Crea descargador basado en datos que funcionan."""
    if not working_data:
        print("❌ No hay datos funcionales para crear descargador")
        return None
    
    print(f"\n🔧 CREANDO DESCARGADOR CON DATOS FUNCIONALES")
    print("=" * 50)
    
    # Seleccionar el mejor conjunto de datos
    best_data = max(working_data, key=lambda x: x.get('total_days', 0))
    
    print(f"📊 Datos seleccionados:")
    print(f"   - Producto: {best_data['product']}")
    print(f"   - Año: {best_data['year']}")
    print(f"   - URL base: {best_data['base_url']}")
    print(f"   - Días disponibles: {best_data.get('total_days', 'N/A')}")
    
    script_content = f'''#!/usr/bin/env python3
"""
Descargador NASA IMERG - Versión Funcional
=========================================

Basado en exploración exitosa de datos disponibles.
Producto: {best_data['product']}
Año: {best_data['year']}
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
        self.base_url = "{best_data['base_url']}"
        self.product = "{best_data['product']}"
        self.year = "{best_data['year']}"
        
        # Configurar sesión con autenticación
        self.session = self._setup_session()
        
        print(f"📥 Descargador configurado:")
        print(f"   - Producto: {{self.product}}")
        print(f"   - Año: {{self.year}}")
        print(f"   - URL base: {{self.base_url}}")
        print(f"   - Directorio: {{self.data_dir}}")
    
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
            print(f"⚠️ Error configurando credenciales: {{e}}")
        
        return session
    
    def get_available_days(self):
        """Obtiene días disponibles."""
        try:
            response = self.session.get(self.base_url, timeout=30)
            if response.status_code == 200:
                import re
                day_dirs = re.findall(r'href="(\\d{{3}})/"', response.text)
                return sorted([int(d) for d in day_dirs])
            else:
                print(f"❌ Error obteniendo días: {{response.status_code}}")
                return []
        except Exception as e:
            print(f"❌ Error: {{e}}")
            return []
    
    def get_files_for_day(self, day_of_year):
        """Obtiene archivos para un día específico."""
        day_url = f"{{self.base_url}}/{{day_of_year:03d}}"
        
        try:
            response = self.session.get(day_url, timeout=15)
            if response.status_code == 200:
                import re
                files = re.findall(r'href="([^"]*\\.HDF5)"', response.text)
                return files
            else:
                return []
        except Exception as e:
            return []
    
    def download_file(self, url, filename):
        """Descarga un archivo."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"⏭️ {{filename}} ya existe")
            return True
        
        try:
            print(f"📥 Descargando {{filename}}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {{filename}} completado ({{size_mb:.1f}} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Error en {{filename}}: {{e}}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_dataset(self, max_files=50):
        """Descarga conjunto de datos."""
        print(f"🚀 DESCARGANDO DATOS {{self.product}} AÑO {{self.year}}")
        print("=" * 50)
        
        # Obtener días disponibles
        available_days = self.get_available_days()
        
        if not available_days:
            print("❌ No se encontraron días disponibles")
            return 0
        
        print(f"📅 Días disponibles: {{len(available_days)}}")
        
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
                    
                url = f"{{self.base_url}}/{{day:03d}}/{{filename}}"
                download_tasks.append((url, filename))
                files_count += 1
        
        print(f"📦 {{len(download_tasks)}} archivos preparados para descarga")
        
        # Ejecutar descargas
        successful = 0
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {{
                executor.submit(self.download_file, url, filename): filename
                for url, filename in download_tasks
            }}
            
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"❌ Excepción en {{filename}}: {{e}}")
                    failed += 1
        
        # Resumen
        print(f"\\n📊 DESCARGA COMPLETADA:")
        print(f"✅ Exitosas: {{successful}}")
        print(f"❌ Fallidas: {{failed}}")
        print(f"📁 Directorio: {{self.data_dir.absolute()}}")
        
        if successful > 0:
            print(f"\\n🎉 ¡{{successful}} archivos descargados!")
            print("🚀 SIGUIENTE PASO:")
            print(f"   python scripts/preprocess_large_dataset.py --raw-dir {{self.data_dir}}")
        
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
'''
    
    # Guardar script
    script_path = Path("scripts/download_nasa_working.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Script funcional guardado: {script_path}")
    return str(script_path)

def main():
    """Función principal."""
    print("🔍 NASA IMERG DATA EXPLORER AVANZADO")
    print("=" * 50)
    
    # Explorar fuentes alternativas
    working_urls = explore_alternative_sources()
    
    # Probar datos específicos
    working_data = test_specific_data_access()
    
    # Resumen
    print(f"\n🎯 RESUMEN DE EXPLORACIÓN:")
    print(f"=" * 30)
    
    if working_data:
        print(f"✅ Datos funcionales encontrados: {len(working_data)}")
        
        for data in working_data:
            print(f"   📊 {data['product']} año {data['year']}: {data.get('total_days', 'N/A')} días")
        
        # Crear descargador funcional
        script_path = create_working_downloader(working_data)
        
        print(f"\n🚀 PRÓXIMOS PASOS:")
        print(f"1. Configurar credenciales: python setup_credentials.py")
        print(f"2. Descargar datos: python {script_path} --max-files 100")
        print(f"3. Procesar datos: python scripts/preprocess_large_dataset.py")
        
    else:
        print(f"💥 NO SE ENCONTRARON DATOS FUNCIONALES")
        print(f"\n💡 SOLUCIONES ALTERNATIVAS:")
        print(f"1. Usar datos locales existentes")
        print(f"2. Descargar datos manualmente desde Giovanni")
        print(f"3. Usar conjunto de datos de prueba más pequeño")
        print(f"4. Contactar soporte NASA GES DISC")

if __name__ == "__main__":
    main()
