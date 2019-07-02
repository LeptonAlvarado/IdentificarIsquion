import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Se crea una funcion para saber cual es el valor mas alto de la matriz
def valorMaxImagen (imagen):
    maxPixel = np.max(imagen)
    minPixel = np.min(imagen)
    meanPixel = np.mean(imagen)
    print(minPixel)
    print(maxPixel)
    print(meanPixel)
    print(imagen)
    return maxPixel


# Se crea una funcion para una funcion para aplicar balance de blancos
def balanceDeBlancos (imagen, maxPixel):
    #imagen2BW = img.imread(imagen) 
    for i in range (461):
        for j in range (460):
            imagen[i][j] = (imagen[i][j]*256)//maxPixel
    return imagen
             
     

# Con este codigo se muestra la imagen tal cual esta
ultSoundOriginal = cv2.imread('prueba.jpg', 0)
'''








