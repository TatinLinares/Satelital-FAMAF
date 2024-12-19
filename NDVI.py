import os
import requests
import re
import numpy as np
import matplotlib.pyplot as plt

from urllib.parse import urljoin
from netCDF4 import Dataset
from helper import calibrate_imag, realce_gama

def get_keys(base_url, prefix):
    """
    Obtains all keys from a public bucket given a prefix.
    """
    url = urljoin(base_url, f"?prefix={prefix}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error accessing {url}: {response.status_code}")
        return []
    
    keys = re.findall(r"<Key>(.*?)</Key>", response.text)
    return keys

def download_file(base_url, key, output_dir):
    """
    Downloads a file from a public S3 URL if it doesn't exist.
    """
    local_filename = os.path.join(output_dir, os.path.basename(key))
    if os.path.exists(local_filename):
        print(f"File {key} already exists. Skipping download.")
        return local_filename

    file_url = urljoin(base_url, key)
    print(f"Downloading {file_url}...")
    
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_filename
    else:
        print(f"Error downloading {file_url}: {response.status_code}")
        return None

def calculate_ndvi(red_path, nir_path):
    """
    Calculates NDVI from red and near-IR bands and returns intermediate data for visualization.
    """
    with Dataset(red_path) as red_ds, Dataset(nir_path) as nir_ds:
        red_full = calibrate_imag(red_ds.variables['Rad'][::2,::2].data, red_ds.variables, 'Ref')
        nir_full = calibrate_imag(nir_ds.variables['Rad'][:].data, nir_ds.variables, 'Ref')

        tamaño = 800
        x0 = 7500
        y0 = 6200
        x1 = x0 + tamaño
        y1 = y0 + tamaño

        red = red_full[y0:y1, x0:x1]
        nir = nir_full[y0:y1, x0:x1]

        # Calculate NDVI
        ndvi = np.where(
            (nir + red) != 0,
            (nir - red) / (nir + red),
            0
        )
        
        return {
            'ndvi': ndvi,
            'red_full': red_full,
            'nir_full': nir_full,
            'red_cropped': red,
            'nir_cropped': nir,
            'crop_coords': (x0, y0, x1, y1)
        }

def visualize_data(data_dict, title_prefix):
    """
    Creates comprehensive visualization of the data including full images,
    cropped region, and NDVI calculation.
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Full resolution red band with crop box
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(data_dict['red_full'], cmap='gray')
    x0, y0, x1, y1 = data_dict['crop_coords']
    rect = plt.Rectangle((y0, x0), y1-y0, x1-x0, fill=False, color='red')
    ax1.add_patch(rect)
    ax1.set_title(f'{title_prefix} - Full Red Band with Crop Region')
    plt.colorbar(im1, ax=ax1)

    # Full resolution NIR band with crop box
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(data_dict['nir_full'], cmap='gray')
    rect = plt.Rectangle((y0, x0), y1-y0, x1-x0, fill=False, color='red')
    ax2.add_patch(rect)
    ax2.set_title(f'{title_prefix} - Full NIR Band with Crop Region')
    plt.colorbar(im2, ax=ax2)

    # Cropped red band
    ax3 = plt.subplot(234)
    im3 = ax3.imshow(data_dict['red_cropped'], cmap='gray')
    ax3.set_title(f'{title_prefix} - Cropped Red Band')
    plt.colorbar(im3, ax=ax3)

    # Cropped NIR band
    ax4 = plt.subplot(235)
    im4 = ax4.imshow(data_dict['nir_cropped'], cmap='gray')
    ax4.set_title(f'{title_prefix} - Cropped NIR Band')
    plt.colorbar(im4, ax=ax4)

    # NDVI result
    ax5 = plt.subplot(236)
    im5 = ax5.imshow(data_dict['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
    ax5.set_title(f'{title_prefix} - NDVI')
    plt.colorbar(im5, ax=ax5)

    # Add histogram of NDVI values
    ax6 = plt.subplot(233)
    ax6.hist(data_dict['ndvi'].flatten(), bins=50, range=(-1, 1))
    ax6.set_title(f'{title_prefix} - NDVI Distribution')
    ax6.set_xlabel('NDVI Value')
    ax6.set_ylabel('Frequency')

    plt.tight_layout()
    return fig

def download_and_process_ndvi(year, day, hour=15):
    """
    Downloads and processes GOES-16 images ffig = plt.figure(figsize=(20, 10))or NDVI calculation.
    """
    BASE_URL = "https://noaa-goes16.s3.amazonaws.com/"
    BASE_PREFIX = "ABI-L1b-RadF"
    OUTPUT_DIR = "goes16_ndvi_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    red_band = "C02"
    nir_band = "C03"
    
    files = {}
    for band in [red_band, nir_band]:
        prefix = f"{BASE_PREFIX}/{year}/{day:03d}/{hour:02d}/OR_ABI-L1b-RadF-M6{band}"
        keys = get_keys(BASE_URL, prefix)
        
        if keys:
            band_dir = os.path.join(OUTPUT_DIR, f"{year}_{day:03d}", band)
            os.makedirs(band_dir, exist_ok=True)
            
            files[band] = download_file(BASE_URL, keys[0], band_dir)
        else:
            print(f"No keys found for {prefix}")
            return None
    
    if all(files.values()):
        return calculate_ndvi(files[red_band], files[nir_band])
    return None


def process_fire_temperature(fire_dir, x0=7500, y0=6200, tamaño=800):
    """
    Procesa las imágenes para crear la composición RGB de temperatura de fuego.
    """
    image_list = sorted(os.listdir(fire_dir))
    images = [Dataset(os.path.join(fire_dir, filename)) for filename in image_list]
    
    # Procesar bandas
    imagen5 = calibrate_imag(images[0].variables['Rad'][::2,::2].data, 
                            images[0].variables, 'Ref')
    imagen6 = calibrate_imag(images[1].variables['Rad'][:].data, 
                            images[1].variables, 'Ref')
    imagen7 = calibrate_imag(images[2].variables['Rad'][:].data, 
                            images[2].variables, 'T')
    
    # Aplicar realces
    red_realce = realce_gama(imagen7, 1, 2.5, 0, 60)
    green_realce = realce_gama(imagen6, 1, 1, 0, 1)
    blue_realce = realce_gama(imagen5, 1, 1, 0, 0.75)
    
    # Crear imagen RGB
    imagen_RGB_fire = np.dstack((red_realce, green_realce, blue_realce))
    
    # Recortar la región de interés
    x0 = int(x0/2)
    y0 = int(y0/2)
    tamaño = int(tamaño/2)
    imagen_RGB_fire_crop = imagen_RGB_fire[y0:y0+tamaño, x0:x0+tamaño]
    
    return {
        'fire_rgb_full': imagen_RGB_fire,
        'fire_rgb_crop': imagen_RGB_fire_crop,
        'crop_coords': (x0, y0, x0+tamaño, y0+tamaño)
    }

def visualize_comparison_with_fire(data_aug, data_dec, fire_data, save_path='ndvi_fire_comparison.png'):
    """
    Creates a comprehensive comparison including NDVI and fire temperature.
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Primera fila: NDVI
    # August NDVI
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(data_aug['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('August 2024 NDVI')
    plt.colorbar(im1, ax=ax1)
    
    # October NDVI
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(data_dec['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_title('October 2024 NDVI')
    plt.colorbar(im2, ax=ax2)
    
    # NDVI Difference
    difference = data_dec['ndvi'] - data_aug['ndvi']
    ax3 = plt.subplot(233)
    im3 = ax3.imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_title('NDVI Difference (October - August)')
    plt.colorbar(im3, ax=ax3, label='NDVI Change')
    
    # Segunda fila: Fire Temperature y análisis adicional
    # Fire Temperature RGB
    ax4 = plt.subplot(234)
    im4 = ax4.imshow(fire_data['fire_rgb_crop'])
    ax4.set_title('September 1st Fire Temperature RGB')
    
    # Máscara de áreas afectadas
    ax5 = plt.subplot(235)
    affected_mask = (difference < -0.2).astype(float)
    im5 = ax5.imshow(affected_mask, cmap='Reds')
    ax5.set_title('Areas with Significant NDVI Decrease')
    plt.colorbar(im5, ax=ax5)
    
    # Histograma de diferencias
    ax6 = plt.subplot(236)
    ax6.hist(difference.flatten(), bins=50, range=(-1, 1), color='blue', alpha=0.7)
    ax6.set_title('NDVI Difference Distribution')
    ax6.set_xlabel('NDVI Change')
    ax6.set_ylabel('Frequency')
    
    # Añadir estadísticas
    stats_text = f"Mean difference: {np.mean(difference):.3f}\n"
    stats_text += f"Std difference: {np.std(difference):.3f}\n"
    stats_text += f"Max decrease: {np.min(difference):.3f}\n"
    stats_text += f"Max increase: {np.max(difference):.3f}\n"
    stats_text += f"Affected area: {np.mean(affected_mask)*100:.1f}%"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle('NDVI and Fire Temperature Analysis', fontsize=16)
    plt.tight_layout()
    return fig

def compare_ndvi_with_fire(fire_dir):
    """
    Realiza una comparación completa incluyendo NDVI y temperatura de fuego.
    """
    # Obtener datos NDVI
    data_aug = download_and_process_ndvi(2024, 213)  # 1 de agosto
    data_oct = download_and_process_ndvi(2024, 275)  # 1 de octubre
    
    # Procesar datos de temperatura de fuego
    fire_data = process_fire_temperature(fire_dir)
    
    if data_aug is not None and data_oct is not None and fire_data is not None:
        # Crear visualizaciones
        print("Creando visualizaciones...")
        fig_aug = visualize_data(data_aug, 'August 2024')
        fig_oct = visualize_data(data_oct, 'October 2024')
        fig_comp = visualize_comparison_with_fire(data_aug, data_oct, fire_data)
        
        # Guardar figuras
        fig_aug.savefig('august_analysis.png', dpi=300, bbox_inches='tight')
        fig_oct.savefig('october_analysis.png', dpi=300, bbox_inches='tight')
        fig_comp.savefig('ndvi_fire_comparison.png', dpi=300, bbox_inches='tight')
        
        # Imprimir estadísticas
        print("\nNDVI Statistics:")
        print(f"August - Mean: {np.mean(data_aug['ndvi']):.3f}, Std: {np.std(data_aug['ndvi']):.3f}")
        print(f"October - Mean: {np.mean(data_oct['ndvi']):.3f}, Std: {np.std(data_oct['ndvi']):.3f}")
        print(f"Difference in means: {np.mean(data_oct['ndvi'] - data_aug['ndvi']):.3f}")
        
        plt.close('all')

if __name__ == "__main__":
    fire_dir = '../goes_fire'  # Directorio con las imágenes de temperatura de fuego
    compare_ndvi_with_fire(fire_dir)