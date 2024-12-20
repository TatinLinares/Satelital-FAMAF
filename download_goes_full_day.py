import os
import re
import requests
from urllib.parse import urljoin
from netCDF4 import Dataset

def validate_nc_files(base_dir):
    """
    Valida todos los archivos NetCDF en el directorio base.

    Args:
        base_dir (str): Directorio base que contiene las carpetas de datos.

    Returns:
        list: Archivos corruptos o no válidos.
    """
    invalid_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Dataset(file_path, 'r'):
                    pass
            except Exception as e:
                print(f"Archivo inválido: {file_path} - {e}")
                invalid_files.append(file_path)

    #Delete invalid files
    for file in invalid_files:
        os.remove(file)

    return invalid_files

def get_keys(base_url, prefix):
    """
    Obtiene todas las claves de un bucket público dado un prefijo.
    
    Args:
        base_url (str): URL base del bucket.
        prefix (str): Prefijo para buscar.

    Returns:
        list: Lista de claves encontradas.
    """
    url = urljoin(base_url, f"?prefix={prefix}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error al acceder a {url}: {response.status_code}")
        return []
    
    # Extraer claves del XML devuelto por S3
    keys = re.findall(r"<Key>(.*?)</Key>", response.text)
    return keys

def download_file(base_url, key, output_dir):
    """
    Descarga un archivo desde una URL pública S3, si no existe previamente.

    Args:
        base_url (str): URL base del bucket.
        key (str): Clave del archivo a descargar.
        output_dir (str): Directorio donde guardar el archivo.
    """
    local_filename = os.path.join(output_dir, os.path.basename(key))
    if os.path.exists(local_filename):
        print(f"El archivo {key} ya existe. Saltando descarga.")
        return

    file_url = urljoin(base_url, key)
    print(f"Descargando {file_url}...")
    
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Error al descargar {file_url}: {response.status_code}")

def main():
    # Configuración
    BASE_URL = "https://noaa-goes16.s3.amazonaws.com/"
    BASE_PREFIX = "ABI-L1b-RadF"
    #BANDS = ["C01", "C02", "C03"]  # Bandas a descargar
    BANDS = ["C05", "C06", "C07"]  # Bandas a descargar
    OUTPUT_DIR = "goes16_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Día actual en formato "YYYY/DDD" (donde DDD es el día del año)
    year = 2024
    day_of_year = 245
    final_ouput_dir = OUTPUT_DIR + f"/{year}/{day_of_year:03d}"
    print("Validando archivos para" + final_ouput_dir)
    validate_nc_files(OUTPUT_DIR)

    for band in BANDS:
        for hour in range(24):
            prefix = f"{BASE_PREFIX}/{year}/{day_of_year:03d}/{hour:02d}/OR_ABI-L1b-RadF-M6{band}"
            print(f"Buscando claves en {prefix}...")
            keys = get_keys(BASE_URL, prefix)

            if keys:
                band_output_dir = os.path.join(final_ouput_dir, f"band_{band}", f"hour_{hour:02d}")
                os.makedirs(band_output_dir, exist_ok=True)

                for key in keys:
                    download_file(BASE_URL, key, band_output_dir)
            else:
                print(f"No se encontraron claves para {prefix}.")

if __name__ == "__main__":
    main()
