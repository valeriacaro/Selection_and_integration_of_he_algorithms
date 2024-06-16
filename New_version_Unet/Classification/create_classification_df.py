import os
from PIL import Image
import pandas as pd
import cv2
import numpy as np

def determinar_clase(mask):
    mask = mask // 85
    valores, conteos = np.unique(mask, return_counts=True)
    
    clase_2 = 0
    clase_3 = 0

    # si hi ha classes tumorals, assignem la majoritària
    if 2 in valores:
        clase_2 = conteos[valores == 2][0]
    if 3 in valores:
        clase_3 = conteos[valores == 3][0]
    
    if clase_2 > 0 or clase_3 > 0:
        if clase_2 >= clase_3:
            return 1
        else:
            return 2
    # si no, retornem classe teixit sa
    else:
        return 0

carpeta_masks = "path_to_masks"

carpeta_imagenes = "path_to_images"

archivos = os.listdir(carpeta_masks)

extensiones_permitidas = ['.jpg', '.jpeg', '.png', '.gif']
imagenes = [archivo for archivo in archivos if os.path.splitext(archivo)[1].lower() in extensiones_permitidas]

df = []

for imagen in imagenes:
    ruta_mask = os.path.join(carpeta_masks, imagen)
    mask = cv2.imread(ruta_mask, 0)
    clase = determinar_clase(mask)
    ruta = os.path.join(carpeta_imagenes, imagen)
    df.append({'Ruta': ruta, 'Clase': clase})
    
df_imagenes = pd.DataFrame(df)

counts = df_imagenes['Clase'].value_counts()

# calculem la distribució total de les dades
percentages = counts / len(df_imagenes) * 100

print("Porcentaje de 0:", percentages.get(0, 0), "%")
print("Porcentaje de 1:", percentages.get(1, 0), "%")
print("Porcentaje de 2:", percentages.get(2, 0), "%")


df_imagenes.to_csv("path_to_save")