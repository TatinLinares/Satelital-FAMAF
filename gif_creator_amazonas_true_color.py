import os
import cv2
import numpy as np
from netCDF4 import Dataset
from helper import calibrate_imag, realce_gama

def create_true_color_video(data_dir, output_video_path, start_hour=10, end_hour=20):
    """
    Crea un video en true color a partir de imágenes calibradas organizadas por bandas y horas.

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
        hour_dir_c01 = os.path.join(data_dir, f"band_C01/hour_{hour:02d}")
        hour_dir_c02 = os.path.join(data_dir, f"band_C02/hour_{hour:02d}")
        hour_dir_c03 = os.path.join(data_dir, f"band_C03/hour_{hour:02d}")

        if not (os.path.exists(hour_dir_c01) and os.path.exists(hour_dir_c02) and os.path.exists(hour_dir_c03)):
            print(f"Datos incompletos para la hora {hour:02d}. Saltando...")
            continue

        # Listar y ordenar archivos por hora
        files_c01 = sorted(os.listdir(hour_dir_c01))
        files_c02 = sorted(os.listdir(hour_dir_c02))
        files_c03 = sorted(os.listdir(hour_dir_c03))

        # Asegurarse de que las listas tengan el mismo número de imágenes
        num_images = min(len(files_c01), len(files_c02), len(files_c03))

        for i in range(num_images):
            dataset_c01 = Dataset(os.path.join(hour_dir_c01, files_c01[i]))
            dataset_c02 = Dataset(os.path.join(hour_dir_c02, files_c02[i]))
            dataset_c03 = Dataset(os.path.join(hour_dir_c03, files_c03[i]))

            # Reducir resolución más agresivamente
            imagen1 = dataset_c01.variables['Rad'][::2, ::2].data  
            imagen2 = dataset_c02.variables['Rad'][::4, ::4].data 
            imagen3 = dataset_c03.variables['Rad'][::2, ::2].data 

            # Calibrar bandas
            imag_calibrate1 = calibrate_imag(imagen1, dataset_c01.variables, 'Ref')  
            imag_calibrate2 = calibrate_imag(imagen2, dataset_c02.variables, 'Ref') 
            imag_calibrate3 = calibrate_imag(imagen3, dataset_c03.variables, 'Ref') 

            # Ajuste gamma
            realce_red = realce_gama(imag_calibrate2, 1, 1, 0, 1)
            realce_green = realce_gama(imag_calibrate3, 1, 1, 0, 1)
            realce_blue = realce_gama(imag_calibrate1, 1, 1, 0, 1)

            # Crear imagen RGB
            filas, columnas = imag_calibrate1.shape
            imagen_RGB_true_color = np.zeros((filas, columnas, 3))
            imagen_RGB_true_color[:, :, 0] = realce_red
            imagen_RGB_true_color[:, :, 1] =  0.1 * realce_green + 0.45 * realce_red + 0.45 * realce_blue
            imagen_RGB_true_color[:, :, 2] = realce_blue

            """ 
            # Recortemos el amazonas
            tamaño = 500
            x0 = 3000
            y0 = 3250
            x1 = x0 + tamaño
            y1 = y0 + tamaño

            imagen_RGB_true_color = imagen_RGB_true_color[x0:x1, y0:y1]
            """
            # Reducir profundidad de color
            imagen_RGB_true_color = (imagen_RGB_true_color * 255).astype(np.uint8)
            imagen_RGB_true_color = (imagen_RGB_true_color // 4 * 4).astype(np.uint8)

            # Inicializar escritor de video
            if out is None:
                height, width = imagen_RGB_true_color.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, 8.0, (width, height))

            # Escribir frame
            out.write(cv2.cvtColor(imagen_RGB_true_color, cv2.COLOR_RGB2BGR))
            frame_count += 1

            print(f"Procesando fotograma {frame_count}")

            # Liberar memoria
            dataset_c01.close()
            dataset_c02.close()
            dataset_c03.close()

    if out is not None:
        out.release()
        print(f"Video guardado en: {output_video_path}")
        print(f"Total de fotogramas: {frame_count}")
    else:
        print("No se generaron suficientes fotogramas para crear un video.")

# Directorio raíz de datos y salida del video
data_dir = "goes16_data/2024/245"
output_video_path = "true_color_animation.mp4"

# Crear el video
create_true_color_video(data_dir, output_video_path)