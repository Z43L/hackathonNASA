#!/usr/bin/env python3
"""
Inspector Detallado de Estructura NASA
=====================================

Para entender exactamente la estructura de directorios NASA.
"""

import requests
import netrc
from pathlib import Path
import re

def inspect_nasa_structure():
    """Inspecciona la estructura detallada de NASA."""
    print("🔍 INSPECTOR DETALLADO DE ESTRUCTURA NASA")
    print("=" * 50)
    
    # Configurar sesión
    session = requests.Session()
    try:
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            auth_info = netrc.netrc()
            username, _, password = auth_info.authenticators('urs.earthdata.nasa.gov')
            session.auth = (username, password)
            print("🔑 Credenciales configuradas")
    except Exception as e:
        print(f"⚠️ Error configurando credenciales: {e}")
    
    # URLs para inspeccionar
    base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
    
    # Inspeccionar año 2022
    year = 2022
    year_url = f"{base_url}/{year}"
    
    print(f"\n🗓️ INSPECCIONANDO AÑO {year}")
    print(f"URL: {year_url}")
    
    try:
        response = session.get(year_url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            
            # Buscar todos los directorios
            dirs = re.findall(r'href="([^"]+)/"', content)
            dirs = [d for d in dirs if d not in ['..', '.']]
            
            print(f"📁 Directorios encontrados: {len(dirs)}")
            for d in dirs[:10]:  # Mostrar primeros 10
                print(f"   - {d}")
            
            if len(dirs) > 10:
                print(f"   ... y {len(dirs) - 10} más")
            
            # Inspeccionar el primer directorio
            if dirs:
                first_dir = dirs[0]
                first_dir_url = f"{year_url}/{first_dir}"
                
                print(f"\n📂 INSPECCIONANDO DIRECTORIO: {first_dir}")
                print(f"URL: {first_dir_url}")
                
                try:
                    dir_response = session.get(first_dir_url, timeout=15)
                    print(f"Status: {dir_response.status_code}")
                    
                    if dir_response.status_code == 200:
                        dir_content = dir_response.text
                        
                        # Buscar archivos
                        files = re.findall(r'href="([^"]*\.HDF5)"', dir_content)
                        other_files = re.findall(r'href="([^"]*\.\w+)"', dir_content)
                        subdirs = re.findall(r'href="([^"]+)/"', dir_content)
                        subdirs = [s for s in subdirs if s not in ['..', '.']]
                        
                        print(f"📄 Archivos HDF5: {len(files)}")
                        for f in files[:5]:
                            print(f"   - {f}")
                        
                        if other_files:
                            print(f"📄 Otros archivos: {len(other_files)}")
                            for f in other_files[:5]:
                                print(f"   - {f}")
                        
                        if subdirs:
                            print(f"📁 Subdirectorios: {len(subdirs)}")
                            for s in subdirs[:5]:
                                print(f"   - {s}")
                            
                            # Inspeccionar primer subdirectorio si existe
                            if subdirs:
                                subdir = subdirs[0]
                                subdir_url = f"{first_dir_url}/{subdir}"
                                
                                print(f"\n📂 INSPECCIONANDO SUBDIRECTORIO: {subdir}")
                                print(f"URL: {subdir_url}")
                                
                                try:
                                    sub_response = session.get(subdir_url, timeout=15)
                                    print(f"Status: {sub_response.status_code}")
                                    
                                    if sub_response.status_code == 200:
                                        sub_content = sub_response.text
                                        sub_files = re.findall(r'href="([^"]*\.HDF5)"', sub_content)
                                        
                                        print(f"📄 Archivos HDF5 en subdirectorio: {len(sub_files)}")
                                        for f in sub_files[:3]:
                                            print(f"   - {f}")
                                            
                                            # Mostrar URL completa del archivo
                                            file_url = f"{subdir_url}/{f}"
                                            print(f"     🔗 URL: {file_url}")
                                    
                                except Exception as e:
                                    print(f"❌ Error en subdirectorio: {e}")
                        
                        # Si no hay archivos ni subdirectorios, mostrar contenido raw
                        if not files and not subdirs:
                            print("\n📋 CONTENIDO RAW (primeras 1000 chars):")
                            print(dir_content[:1000])
                    
                except Exception as e:
                    print(f"❌ Error inspeccionando directorio: {e}")
        
        else:
            print(f"❌ Error accediendo al año: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error general: {e}")

def test_alternative_products():
    """Prueba productos alternativos que podrían tener mejor estructura."""
    print("\n🧪 PROBANDO PRODUCTOS ALTERNATIVOS")
    print("=" * 40)
    
    session = requests.Session()
    try:
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            auth_info = netrc.netrc()
            username, _, password = auth_info.authenticators('urs.earthdata.nasa.gov')
            session.auth = (username, password)
    except:
        pass
    
    # Productos alternativos para probar
    products = [
        'GPM_3IMERGHH.07',  # Half-hourly
        'GPM_3IMERGM.07',   # Monthly
        'GPM_3IMERGDE.07',  # Daily early
    ]
    
    base_nasa_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3"
    year = 2022
    
    for product in products:
        print(f"\n--- {product} ---")
        product_url = f"{base_nasa_url}/{product}/{year}"
        
        try:
            response = session.get(product_url, timeout=15)
            print(f"URL: {product_url}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                content = response.text
                
                # Buscar estructura
                dirs = re.findall(r'href="([^"]+)/"', content)
                dirs = [d for d in dirs if d not in ['..', '.']]
                
                files = re.findall(r'href="([^"]*\.HDF5)"', content)
                
                print(f"📁 Directorios: {len(dirs)}")
                print(f"📄 Archivos HDF5 directos: {len(files)}")
                
                if files:
                    print(f"✅ ¡Archivos encontrados directamente!")
                    for f in files[:3]:
                        print(f"   - {f}")
                        
                elif dirs:
                    # Probar primer directorio
                    first_dir_url = f"{product_url}/{dirs[0]}"
                    try:
                        dir_response = session.get(first_dir_url, timeout=10)
                        if dir_response.status_code == 200:
                            dir_files = re.findall(r'href="([^"]*\.HDF5)"', dir_response.text)
                            print(f"📄 Archivos en {dirs[0]}: {len(dir_files)}")
                            
                            if dir_files:
                                print(f"✅ ¡Archivos encontrados en subdirectorio!")
                                for f in dir_files[:2]:
                                    print(f"   - {f}")
                    except:
                        pass
                        
            else:
                print(f"❌ No accesible")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Función principal."""
    inspect_nasa_structure()
    test_alternative_products()

if __name__ == "__main__":
    main()