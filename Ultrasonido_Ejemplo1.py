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

ret,thresh1 = cv2.threshold(ultSoundOriginal,0,255,cv2.THRESH_BINARY)
cv2.imshow('Umbral', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
laplacian = cv2.Laplacian(ultSoundOriginal,cv2.CV_64F)
sobelx = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,0,1,ksize=3)

cv2.imshow('La placiana', laplacian)
cv2.imshow('Sobel x', sobelx)
cv2.imshow('Sobel y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
# X
sobelx64f = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8x = np.uint8(abs_sobel64f)
# Y
sobelx64f = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,0,1,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8y = np.uint8(abs_sobel64f)

cv2.imshow('Sobel x x64', sobel_8x)
cv2.imshow('Sobel y x64', sobel_8y)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
_,th = cv2.threshold(sobel_8y,127,255,0)
imagen, contornos, jerarquia = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sobel_8y, contornos, -1, (0,255,0), 3)
cv2.imshow('Sobel y x64', sobel_8y)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
#valorMaxImagen (ultSoundOriginal)
'''
# Cany

bordes = cv2.Canny(ultSoundOriginal,0,100)
cv2.imshow('Caby', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
