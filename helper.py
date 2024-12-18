import numpy as np

def calibrate_imag(imagen, metadato, U='T'):
    """
    Calibra la imagen basada en los metadatos de GOES-16.

    Args:
        imagen (ndarray): Datos de la imagen sin procesar.
        metadato (dict): Metadatos asociados con la imagen.
        U (str): Unidad de salida ('T' para temperatura, 'Rad' para radiancia, 'Ref' para reflectancia).

    Returns:
        ndarray: Imagen calibrada.
    """
    canal = int(metadato['band_id'][:])
    if canal >= 7 and U == 'T':
        # Calibración a temperatura de brillo
        fk1 = metadato['planck_fk1'][0]
        fk2 = metadato['planck_fk2'][0]
        bc1 = metadato['planck_bc1'][0]
        bc2 = metadato['planck_bc2'][0]

        imagen = np.where(imagen > 0, imagen, np.nan) 
        imag_cal = (fk2 / (np.log((fk1 / imagen) + 1)) - bc1) / bc2 - 273.15 
    elif U == 'Rad':
        # Calibración a radiancia
        pendiente = metadato['Rad'].scale_factor
        ordenada = metadato['Rad'].add_offset
        imag_cal = imagen * pendiente + ordenada
    elif U == 'Ref':
        # Calibración a reflectancia
        kapa0 = metadato['kappa0'][0].data
        imag_cal = kapa0 * imagen
    else:
        raise ValueError(f"Unidad de calibración '{U}' no reconocida. Use 'T', 'Rad' o 'Ref'.")

    return imag_cal

def realce_gama(V, A, gamma, Vmin, Vmax):
    """
    Aplica el realce de gamma a una imagen o matriz de datos.

    Args:
        V (ndarray): Matriz de datos de entrada.
        A (float): Amplitud de salida.
        gamma (float): Exponente de gamma para el realce.
        Vmin (float): Valor mínimo para la normalización.
        Vmax (float): Valor máximo para la normalización.

    Returns:
        ndarray: Matriz con el realce de gamma aplicado.
    """
    Vaux = np.clip((V - Vmin) / (Vmax - Vmin), 0, 1)
    Vout = A * Vaux**gamma
    return Vout
