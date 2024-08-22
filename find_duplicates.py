import numpy as np
import os
import pandas as pd

from difflib import SequenceMatcher
from geopy.distance import geodesic
from joblib import Parallel, delayed
from tqdm import tqdm

df = pd.read_csv('data/2024-02-febrero.csv', dialect='excel', keep_default_na=False, dtype=str)

# Función para convertir cadenas de texto a números
def convertir_a_numero(valor):
    try:
        return float(valor.replace(',', '').replace(' ', '').replace('ha', '').replace('m2', '').strip())
    except (ValueError, AttributeError):
        return np.nan

# Función para convertir superficie a metros cuadrados
def convertir_superficie(superficie, unidad):
    if unidad == 'm²':
        return superficie
    elif unidad == 'ha':
        return superficie * 10000  # 1 hectárea = 10,000 m²
    else:
        return np.nan  # Valor faltante o desconocido

# Función para calcular la similaridad de cadenas
def similar(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0  # Si alguno es NA, no son similares
    return SequenceMatcher(None, a, b).ratio()

def es_duplicado(fila1, fila2):
    # 1. Verificación estricta de coordenadas
    try:
        coord1 = (float(fila1['latitude']), float(fila1['longitude']))
        coord2 = (float(fila2['latitude']), float(fila2['longitude']))
        distancia = geodesic(coord1, coord2).meters
    except (ValueError, TypeError):
        distancia = float('inf')  # Asumir que están muy lejos si las coordenadas no son válidas

    if distancia > 200:  # Umbral de distancia más estricto
        return False

    # 2. Comparación de tipo de propiedad
    if fila1['property_type'] != fila2['property_type']:
        return False

    # 3. Comparación de descripción y características clave
    similitud_descripcion = similar(fila1['description'], fila2['description'])

    if similitud_descripcion < 0.7:  # Ajuste a un umbral más alto para la descripción
        return False

    # 4. Comparación de superficies y unidades
    superficie1 = convertir_superficie(convertir_a_numero(fila1['total_surface']), fila1['total_surface_unit'])
    superficie2 = convertir_superficie(convertir_a_numero(fila2['total_surface']), fila2['total_surface_unit'])

    if superficie1 is None or superficie2 is None or abs(superficie1 - superficie2) > 10:  # Tolerancia de 10 m²
        return False

    # Si pasa todas las verificaciones anteriores, se considera un duplicado
    return True

def get_duplicates_for_idx(idx):
    if os.path.isfile(f'output/{idx:06}.log'):
        with open(f'output/{idx:06}.log', 'r') as f:
            duplicated_idx = f.readlines()[0]
        return duplicated_idx
    idxs = list(range(len(df)))
    idxs.remove(idx)
    duplicated_idx = str(idx)
    dup = [idx]
    for i in idxs:
        if es_duplicado(df.iloc[idx], df.iloc[i]):
            duplicated_idx += f' {str(i)}'
            dup.append(i)
    for x in dup:
        with open(f'output/{x:06}.log', 'w') as f:
            f.write(f'{duplicated_idx}')
    return duplicated_idx

duplicated = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(get_duplicates_for_idx)(idx) for idx, _ in zip(list(range(len(df))), tqdm(list(range(len(df))))))

df['duplicated'] = duplicated

df.to_csv('data/2024-02-febrero_dup_identified.csv',index=False)

