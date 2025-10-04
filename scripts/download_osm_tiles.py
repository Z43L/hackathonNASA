import os
import requests
from tqdm import tqdm

# Configuración
ZOOM_MIN = 0  # Zoom mínimo
ZOOM_MAX = 3  # Zoom máximo (aumenta para más detalle, pero descarga más datos)
TILE_SERVER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OUTPUT_DIR = "public/tiles/earth"

def download_tiles():
    for z in range(ZOOM_MIN, ZOOM_MAX + 1):
        n = 2 ** z
        print(f"Descargando zoom {z} ({n}x{n} tiles)...")
        for x in tqdm(range(n), desc=f"Zoom {z} - X"):
            for y in range(n):
                url = TILE_SERVER.format(z=z, x=x, y=y)
                out_dir = os.path.join(OUTPUT_DIR, str(z), str(x))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{y}.png")
                if os.path.exists(out_path):
                    continue  # Ya descargado
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(r.content)
                    else:
                        print(f"Tile no disponible: {url}")
                except Exception as e:
                    print(f"Error descargando {url}: {e}")

if __name__ == "__main__":
    download_tiles()
    print("Descarga finalizada. Ahora puedes usar los tiles localmente.")
