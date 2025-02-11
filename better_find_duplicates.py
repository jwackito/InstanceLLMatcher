import numpy as np
import os
import pandas as pd

from difflib import SequenceMatcher
from geopy.distance import geodesic
from joblib import Parallel, delayed
from tqdm import tqdm


# Función para convertir cadenas de texto a números
def convertir_a_numero(valor):
    try:
        return float(valor.replace(',', '.').replace(' ', '').replace('ha', '').replace('m2', '').strip())
    except (ValueError, AttributeError):
        try:
            #print(f'U shouldn\t have entered here... "{valor}"')
            return float(valor.values[0].replace(',', '.').replace(' ', '').replace('ha', '').replace('m2', '').strip())
        except:
            return 0.

# Función para convertir superficie a metros cuadrados
def convertir_superficie(superficie, unidad):
    if unidad == 'm²' or unidad == 'm2':
        return superficie.__trunc__()
    elif unidad == 'ha':
        return (superficie * 10000).__trunc__()  # 1 hectárea = 10,000 m²
    else:
        #print(f'U shouldn\t have entered here... unidad: "{unidad}" - superficie: "{superficie}"')
        return 0  # Valor faltante o desconocido

# Función para calcular la similaridad de cadenas
def similar(a, b):
    if pd.isna(a) or pd.isna(b):
        return 1  # Si alguno es NA, no son similares
    return SequenceMatcher(None, a, b).ratio()

def comparar_superficie(fila1, fila2):
    if abs(fila1.uniform_surface - fila2.uniform_surface) > 10:
        return False
    else:
        return True

def comparar_precio(fila1, fila2):
    price1 = convertir_a_numero(fila1.price)
    price2 = convertir_a_numero(fila2.price)
    # Si uno de los precios es nan, no se puede saber si son diferentes por el precio
    if price1 is np.nan or price2 is np.nan:
        return True

    if abs(price1 - price2) > 1000:
        return False
    else:
        return True

def es_duplicado(fila1, fila2):
    # distinto property_group, son diferentes inmuebles
    #if fila1['property_group'] != fila2['property_group']:
    #    return False

    # Verificación estricta de coordenadas
    #try:
    #    coord1 = (float(fila1['latitude']), float(fila1['longitude']))
    #    coord2 = (float(fila2['latitude']), float(fila2['longitude']))
    #    distancia = geodesic(coord1, coord2).meters
    #except (ValueError, TypeError):
    #    distancia = float('inf')  # Asumir que están muy lejos si las coordenadas no son válidas

    #if distancia > 200:  # Umbral de distancia más estricto
    #    return False

    # 2. Comparación de tipo de propiedad
    #if fila1['property_type'] != fila2['property_type']:
    #    return False

    # 3. Comparación de descripción y características clave
    similitud_descripcion = similar(fila1['description'], fila2['description'])

    if similitud_descripcion < 0.7:  # Ajuste a un umbral más alto para la descripción
        return False

    # 4. Comparación de superficies y unidades
    if not comparar_superficie(fila1, fila2):
        return False
    
    #superficie1 = convertir_superficie(convertir_a_numero(fila1['total_surface']), fila1['total_surface_unit'])
    #superficie2 = convertir_superficie(convertir_a_numero(fila2['total_surface']), fila2['total_surface_unit'])

    #if superficie1 is None or superficie2 is None or abs(superficie1 - superficie2) > 10:  # Tolerancia de 10 m²
    #    return False

    if not comparar_precio(fila1, fila2):
        return False

    # Si pasa todas las verificaciones anteriores, se considera un duplicado
    return True

def process_indices(indices):
    cut = df.iloc[indices]
    duplicated = []
    for idx in indices:
        dupidx, dupuid = get_duplicates_for_idx(idx, indices, cut)
        duplicated.append((int(idx), dupidx, dupuid))
    return duplicated

def get_duplicates_for_idx(idx, indices, df):
    idxs = list(indices)
    idxs.remove(idx)
    #duplicated_idx = idx.astype(list)
    duplicated_idx = []
    duplicated_uid = []
    for i in idxs:
        if es_duplicado(df.loc[idx], df.loc[i]):
            duplicated_idx.append(int(i))
            duplicated_uid.append(df.loc[i].uid)
    return duplicated_idx, duplicated_uid

def process_uids(uids):
    cut = df[df.uid.isin(uids)]
    duplicated = []
    for uid in uids:
        dupidx, dupuid = get_duplicates_for_uid(uid, uids, cut)
        duplicated.append((uid, dupidx, dupuid))
    return duplicated

def get_duplicates_for_uid(uid, uids, df):
    alluids = list(uids)
    #alluids.remove(uid)
    #duplicated_idx = idx.astype(list)
    duplicated_idx = []
    duplicated_uid = []
    for i in alluids:
        if es_duplicado(df[df.uid == uid].iloc[0], df[df.uid == i].iloc[0]):
            duplicated_idx.append(i)
            duplicated_uid.append(i)
        else:
            print(f'{uid} not equal to {i}')
    return duplicated_idx, duplicated_uid


def get_uniform_surface(fila):
    # Convertir todo a m²
    ts = convertir_superficie(convertir_a_numero(fila.total_surface), fila.total_surface_unit)
    cs = convertir_superficie(convertir_a_numero(fila.covered_surface), fila.covered_surface_unit)
    us = convertir_superficie(convertir_a_numero(fila.uncovered_surface), fila.uncovered_surface_unit)
    ls = convertir_superficie(convertir_a_numero(fila.land_surface), fila.land_surface_unit)
    return max([ts, cs, us, ls])
    if ts != 0:
        return ts
    if cs != 0:
        if us != 0:
            return cs + us
        else:
            return cs
    if us != 0:
        return us
    if ls != 0:
        return ls
    # print(f'Can\'t get uniform surface for unit {fila}')
    return 0


print('Loading dataset...')
#df = pd.read_csv('input/2024/07. inmo_jul_24.zip', dialect='excel', keep_default_na=True, dtype=str).fillna('')
#df = pd.read_csv('input/2024/inmo_julio24.zip', dialect='excel', keep_default_na=True, dtype=str).fillna('')
#df = pd.read_csv('input/2024/inmo_2024-02-01_23-41-27.csv', dialect='excel', keep_default_na=True, dtype=str).fillna('')
df = pd.read_csv('input/2024/inmo_2022-06-16_09-02-42_arba_code.zip', dialect='excel', keep_default_na=True, dtype=str).fillna('')
df['latlon'] = df.latitude.apply(lambda x: x[:8]) + '__' + df.longitude.apply(lambda x: x[:8])
#df['uid'] = df.site_abbreviation + df.listing_id.apply(lambda x: x if not x.startswith('MLA') else x[3:])

df['uid'] = df.listing_id
surfaces = []
for fila in df.itertuples():
    surfaces.append(get_uniform_surface(fila))
df['uniform_surface'] = surfaces

print(f'Grouping')
grouping = df.groupby(['property_group', 'arba_code', 'latlon'])
#grouping = df.groupby(['latlon', 'property_group', 'uniform_surface', 'price'])

print(f'Max grouping {grouping.address.count().max()}')
chk_idx = []
for k in grouping.indices:
    v = grouping.indices[k]
    chk_idx.append(v)


#df.to_csv('output/2024/07. inmo_jul_24_arba-code.zip', index=False)
#### pre procesamiento de GROUND TRUTH duplicados
print('Computing stats')
gt = pd.read_excel('input/2024/duplicados_curados_20220616_subgrupo1.xlsx')
#gt = pd.read_excel('input/2024/duplicados_curados_20240201_subgrupo1.xlsx')

uiddic = {uid:0 for uid in df.uid.values}
uidset = set()
def process_line(s):
    uids = []
    dups = []
    if s.startswith(';'):
        s = s[1:]
    if s.endswith(';'):
        s = s[:-1]
    flatuids = []
    for x in s.split(';'):
        if x == '':
            continue
        if x in uiddic:
            uids.append(x)
            uidset.add(x)
            dups.append([y for y in s.split(';') if (y != x) and y in uiddic])
            for uid in [y for y in s.split(';') if (y != x) and y in uiddic]:
                uidset.add(uid)
            flatuids.append(x)
    return [uids,dups,flatuids]

l = []
flat = []
for x in gt.values:
    uids, dups, flatuids = process_line(x[0])
    for i,j in zip(uids, dups):
        l.append([i,j])
    flat.append(flatuids)

gt = pd.DataFrame(l, columns=['uid', 'dupuid'])

print(f'Finding duplicates')
#duplicated = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(process_indices)(idx) for idx, _ in zip(chk_idx, tqdm(list(range(len(chk_idx))))))
#dups = [0] * len(df)
#dupidx_result = [''] * len(df)
#dupuid_result = [''] * len(df)
#for d in duplicated:
#    for idx, dupidx, dupuid in d:
#        dups[idx] = len(dupidx)
#        dupidx_result[idx] = dupidx
#        dupuid_result[idx] = dupuid
#
#df['n_dups'] = dups
#df['dupidx'] = dupidx_result
#df['dupuid'] = dupuid_result

duplicated = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(process_uids)(uids) for uids, _ in zip(flat, tqdm(list(range(len(flat))))))
uid2idx = {uid: idx for idx, uid in enumerate(df.uid.values)}
dups = [0] * len(df)
dupidx_result = [''] * len(df)
dupuid_result = [''] * len(df)
for d in duplicated:
    for uid, dupidx, dupuid in d:
        try:
            idx = uid2idx[uid]
            dups[idx] = len(dupidx)
            dupidx_result[idx] = dupidx
            dupuid_result[idx] = dupuid
        except KeyError:
            continue

df['n_dups'] = dups
df['dupidx'] = dupidx_result
df['dupuid'] = dupuid_result

#### Performance measurement

# check only the uids that are present in the df AND in the gt
checkuids = []
for uid in df.uid.values:
    if uid in gt.uid.values:
        checkuids.append(uid)
print('Computing stats for ', len(checkuids), ' uids.')

def count_hits_miss(uid):
    dfdups = set(df[df.uid == uid].dupuid.values[0])
    gtdups = set(gt[gt.uid == uid].dupuid.values[0])
    #print(f'{uid}: {dfdups} -- {gtdups}')
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for uid in dfdups:
        if uid not in checkuids:
            continue
        if uid in gtdups:
            TP += 1
        else:
            if uid in gt.uid.values:
                FP += 1
    # Cuenta como FP el uid que es duplicado de sí mismo.
    FP -= 1
    for uid in gtdups:
        if uid not in checkuids:
            continue
        if (uid not in dfdups):# and (uid in df.uid.values):
            FN += 1
    TN = len(uidset) - (TP + FN)
    return [TP, TN, FP, FN]

#results = []
#for uid in checkuids:
#    results.append(count_hits_miss(uid))

results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(count_hits_miss)(uid) for uid, _ in zip(checkuids, tqdm(list(range(len(checkuids))))))

res = pd.DataFrame(checkuids, columns=['uid'])
res[['TP', 'TN', 'FP', 'FN']] = results
res['acc'] = (res.TP+res.TN)/(res.TP+res.TN+res.FP+res.FN)
res['pre'] = res.TP/(res.TP+res.FP)
res['rec'] = res.TP/(res.TP+res.FN)
res['f_1'] = (2*res.TP) / (2*res.TP+res.FP+res.FN)

acc = (res.TP.sum()+res.TN.sum())/(res.TP.sum()+res.TN.sum()+res.FP.sum()+res.FN.sum())
pre = res.TP.sum()/(res.TP.sum()+res.FP.sum()) 
rec = res.TP.sum()/(res.TP.sum()+res.FN.sum())
f1 = 2*((pre*rec)/(pre+rec))


print(f'Accuracy = {acc}')
print(f'Precision = {pre}')
print(f'Recall = {rec}')
print(f'F1 = {f1}')

print(f'TP = {res.TP.sum()}')
print(f'FP = {res.FP.sum()}')
print(f'FN = {res.FN.sum()}')

