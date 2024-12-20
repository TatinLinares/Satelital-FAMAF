import os
import cv2
import numpy as np
from netCDF4 import Dataset
from helper import calibrate_imag, realce_gama

def create_fire_temp_rgb_video(data_dir, output_video_path, start_hour=00, end_hour=24):
    """
    Crea un video RGB de temperatura de fuego a partir de imágenes calibradas de las bandas 5, 6 y 7.

    Args:
        data_dir (str): Directorio raíz que contiene las bandas organizadas por horas.
        output_video_path (str): Ruta para guardar el video resultante.
        start_hour (int): Hora inicial de procesamiento.
        end_hour (int): Hora final de procesamiento.
    """
    out = None
    frame_count = 0

    for hour in range(start_hour, end_hour + 1):
        print(f"Procesando hora {hour:02d}...")
        hour_dir_c05 = os.path.join(data_dir, f"band_C05/hour_{hour:02d}")
        hour_dir_c06 = os.path.join(data_dir, f"band_C06/hour_{hour:02d}")
        hour_dir_c07 = os.path.join(data_dir, f"band_C07/hour_{hour:02d}")

        if not (os.path.exists(hour_dir_c05) and os.path.exists(hour_dir_c06) and os.path.exists(hour_dir_c07)):
            print(f"Datos incompletos para la hora {hour:02d}. Saltando...")
            continue

        # Listar y ordenar archivos por hora
        files_c05 = sorted(os.listdir(hour_dir_c05))
        files_c06 = sorted(os.listdir(hour_dir_c06))
        files_c07 = sorted(os.listdir(hour_dir_c07))

        # Asegurarse de que las listas tengan el mismo número de imágenes
        num_images = min(len(files_c05), len(files_c06), len(files_c07))

        for i in range(num_images):
            dataset_c05 = Dataset(os.path.join(hour_dir_c05, files_c05[i]))
            dataset_c06 = Dataset(os.path.join(hour_dir_c06, files_c06[i]))
            dataset_c07 = Dataset(os.path.join(hour_dir_c07, files_c07[i]))

            # Reducir resolución 
            imagen5 = dataset_c05.variables['Rad'][::2, ::2].data
            imagen6 = dataset_c06.variables['Rad'][:].data
            imagen7 = dataset_c07.variables['Rad'][:].data

            # Calibrar bandas
            imag_calibrate5 = calibrate_imag(imagen5, dataset_c05.variables, 'Ref')
            imag_calibrate6 = calibrate_imag(imagen6, dataset_c06.variables, 'Ref')
            imag_calibrate7 = calibrate_imag(imagen7, dataset_c07.variables, 'T')

            # Crear imagen RGB para temperatura de fuego
            filas, columnas = imag_calibrate5.shape
            imagen_RGB_fire = np.zeros((filas, columnas, 3))

            # Realce de canales con parámetros específicos para Fire Temperature RGB
            realce_red = realce_gama(imag_calibrate7, 1, 2.5, 0, 60)
            realce_green = realce_gama(imag_calibrate6, 1, 1, 0, 1)
            realce_blue = realce_gama(imag_calibrate5, 1, 1, 0, 0.75)

            imagen_RGB_fire[:,:,0] = realce_red
            imagen_RGB_fire[:,:,1] = realce_green
            imagen_RGB_fire[:,:,2] = realce_blue

            """
            # Recortemos el amazonas
            tamaño = 1000
            x0 = 3000
            y0 = 3250
            x1 = x0 + tamaño
            y1 = y0 + tamaño

            imagen_RGB_fire = imagen_RGB_fire[x0:x1, y0:y1] 
            """
            # Reducir profundidad de color
            imagen_RGB_fire = (imagen_RGB_fire * 255).astype(np.uint8)
            imagen_RGB_fire = (imagen_RGB_fire // 4 * 4).astype(np.uint8)

            # Inicializar escritor de video
            if out is None:
                height, width = imagen_RGB_fire.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, 8.0, (width, height))

            # Escribir frame
            out.write(cv2.cvtColor(imagen_RGB_fire, cv2.COLOR_RGB2BGR))
            frame_count += 1

            print(f"Procesando fotograma {frame_count}")

            # Liberar memoria
            dataset_c05.close()
            dataset_c06.close()
            dataset_c07.close()

    if out is not None:
        out.release()
        print(f"Video guardado en: {output_video_path}")
        print(f"Total de fotogramas: {frame_count}")
    else:
        print("No se generaron suficientes fotogramas para crear un video.")

# Directorio raíz de datos y salida del video
data_dir = "goes16_data/2024/245"
output_video_path = "fire_temp_rgb_animation.mp4"

# Crear el video
create_fire_temp_rgb_video(data_dir, output_video_path)