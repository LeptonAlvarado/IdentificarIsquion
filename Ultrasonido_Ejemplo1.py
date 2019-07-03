import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Se crea una funcion para saber caracteristicas de la matriz
def valorMaxImagen (imagen):
    maxPixel = np.max(imagen)
    minPixel = np.min(imagen)
    meanPixel = np.mean(imagen)
    print(minPixel)
    print(maxPixel)
    print(meanPixel)
    print(imagen)
    return maxPixel   

# Con este codigo se lee la imagen
ultSoundOriginal = cv2.imread('prueba.jpg', 0)

 # Se aplica un umbral en el que si es diferente de 0 se haga 1
 # https://www.pyimagesearch.com/2014/09/08/thresholding-simple-image-segmentation-using-opencv/
ret,umbralUlt  = cv2.threshold(ultSoundOriginal,0,255,cv2.THRESH_BINARY)
cv2.imshow('Umbral', umbralUlt )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se hara un cierre de los blancos para posteriormente detectar bordes
kernel = np.ones((3,3),np.uint8)
cierre = cv2.morphologyEx(umbralUlt, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Cierre', cierre )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Deteccion de borde
contours, hierarchy = cv2.findContours(cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ultSoundOriginal, contours, -1, (255, 0, 0), 2)
cv2.imshow('Contornos', cierre )
cv2.imshow('dx', ultSoundOriginal )
cv2.waitKey(0)
cv2.destroyAllWindows()

print(contours)

'''
laplacian = cv2.Laplacian(ultSoundOriginal,cv2.CV_64F)
sobelx = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(ultSoundOriginal,cv2.CV_64F,0,1,ksize=3)

cv2.imshow('La placiana', laplacian)
cv2.imshow('Sobel x', sobelx)
cv2.imshow('Sobel y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows();

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

_,th = cv2.threshold(sobel_8y,127,255,0)
imagen, contornos, jerarquia = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sobel_8y, contornos, -1, (0,255,0), 3)
cv2.imshow('Sobel y x64', sobel_8y)
cv2.waitKey(0)
cv2.destroyAllWindows();

#valorMaxImagen (ultSoundOriginal)

# Cany

bordes = cv2.Canny(ultSoundOriginal,0,100)
cv2.imshow('Caby', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows();
'''
