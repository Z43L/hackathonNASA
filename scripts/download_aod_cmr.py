import requests
import h5py
import os
from datetime import datetime
import getpass

def search_cmr_granules(short_name, version, start_date, end_date, bbox, max_results=5):
    """
    Busca archivos en CMR para un producto y rango de fechas.
    """
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": bbox,
        "page_size": max_results
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    results = resp.json()["feed"]["entry"]
    return [l["href"] for r in results for l in r["links"] if l["href"].endswith(".hdf")]

def download_file(url, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, url.split("/")[-1])
    # Solo HTTP/HTTPS
    from requests.auth import HTTPBasicAuth
    import netrc
    try:
        info = netrc.netrc()
        username, _, password = info.authenticators("urs.earthdata.nasa.gov")
    except Exception:
        username = input("Earthdata username: ")
        password = getpass.getpass("Earthdata password: ")
    with requests.get(url, stream=True, auth=HTTPBasicAuth(username, password)) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path


# Configuración de variables y factores de escala por producto
PRODUCT_CONFIG = {
    # MODIS AOD
    "MCD19A2": {
        "variable": "Optical_Depth_055",
        "scale": 0.001,
        "fill": -9999
    },
    # MODIS Terra AOD
    "MOD04_L2": {
        "variable": "Optical_Depth_Land_And_Ocean",
        "scale": 0.001,
        "fill": -9999
    },
    # MODIS Aqua AOD
    "MYD04_L2": {
        "variable": "Optical_Depth_Land_And_Ocean",
        "scale": 0.001,
        "fill": -9999
    },
    # OMI NO2
    "OMNO2d": {
        "variable": "ColumnAmountNO2",
        "scale": 1.0,
        "fill": -1.267651e+30
    },
    # OMI SO2
    "OMSO2e": {
        "variable": "ColumnAmountSO2",
        "scale": 1.0,
        "fill": -1.267651e+30
    },
    # OMI O3
    "OMO3PR": {
        "variable": "OzoneProfile",
        "scale": 1.0,
        "fill": -1.267651e+30
    },
    # TROPOMI NO2
    "S5P_OFFL_L2__NO2____": {
        "variable": "nitrogendioxide_tropospheric_column",
        "scale": 1.0e5,
        "fill": -9999
    },
    # TROPOMI SO2
    "S5P_OFFL_L2__SO2____": {
        "variable": "sulphur_dioxide_total_vertical_column",
        "scale": 1.0,
        "fill": -9999
    },
    # TROPOMI CO
    "S5P_OFFL_L2__CO_____": {
        "variable": "carbonmonoxide_total_column",
        "scale": 1.0,
        "fill": -9999
    },
    # TROPOMI O3
    "S5P_OFFL_L2__O3_____": {
        "variable": "ozone_total_vertical_column",
        "scale": 1.0,
        "fill": -9999
    },
}

def get_latest_version(short_name):
    """Obtiene la versión más reciente disponible para un producto usando la API de CMR."""
    url = "https://cmr.earthdata.nasa.gov/search/collections.json"
    params = {"short_name": short_name}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    entries = resp.json()["feed"]["entry"]
    if not entries:
        raise ValueError(f"No se encontró ninguna colección para {short_name}")
    # Devuelve la versión más alta encontrada
    versions = [e["version_id"] for e in entries if "version_id" in e]
    return sorted(versions)[-1] if versions else None

def read_variable_from_hdf(hdf_path, product):
    cfg = PRODUCT_CONFIG.get(product)

def main():
        # Cambia estos parámetros para el producto que desees
        short_name = "MCD19A2"  # Cambia a OMNO2d, OMSO2e, etc. según el producto
        version = get_latest_version(short_name)
        start_date = "2023-05-15"
        end_date = "2023-05-15"
        bbox = "-4.0,40.0,-3.5,40.5"  # min lon, min lat, max lon, max lat
        print(f"Buscando archivos para {short_name} versión {version}...")
        urls = search_cmr_granules(short_name, version, start_date, end_date, bbox)
        print(f"Archivos encontrados: {urls}")
        import h5py
        for url in urls:
            if not (url.startswith("http://") or url.startswith("https://")):
                print(f"Ignorando URL no soportada: {url}")
                continue
            print(f"Descargando {url}...")
            local = download_file(url)
            # Validar si es un archivo HDF5 válido
            is_hdf5 = False
            try:
                with h5py.File(local, 'r') as f:
                    is_hdf5 = True
            except Exception as e:
                print(f"ADVERTENCIA: El archivo {local} no es un HDF5 válido. Probablemente es un HTML de error o está corrupto.\nError: {e}")
                # Guardar el contenido para depuración
                try:
                    with open(local, 'rb') as f:
                        content = f.read()
                    # Si parece HTML, guardar como .html
                    if b'<html' in content[:1000].lower():
                        html_path = local + '.html'
                        with open(html_path, 'wb') as f:
                            f.write(content)
                        print(f"Se ha guardado el contenido HTML de error en: {html_path}")
                except Exception as e2:
                    print(f"No se pudo guardar el contenido de error: {e2}")
            if is_hdf5:
                print(f"Leyendo datos de {local}...")
                data = read_variable_from_hdf(local, short_name)
                print(f"{short_name} shape: {data.shape}, valores: min={data.min()}, max={data.max()}")
            else:
                print(f"Omitiendo archivo no válido: {local}")

if __name__ == "__main__":
    main()
