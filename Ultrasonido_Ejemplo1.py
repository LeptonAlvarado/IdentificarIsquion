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

########################### Obtaining the image ##########################
# Si la imagen esta en la misma carpeta que el programa se puede poner solo el nombre de la imagen
# Si no, se necesita poner la direccion
# cv2.imread('imagen',bandera)
# bandera = 0 Escala de grises, = 1 A Color, = -1 Carga la imagen como tal, incluyendo el canal alfa
ultSoundOriginal = cv2.imread('C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/IT SandraITXXXX0E_Frame172.jpg', 0)

# Se aplica un umbral en el que si es diferente de 0 se haga 1
# https://www.pyimagesearch.com/2014/09/08/thresholding-simple-image-segmentation-using-opencv/
ret,umbralUlt  = cv2.threshold(ultSoundOriginal,0,255,cv2.THRESH_BINARY)
#cv2.imshow('Umbral', umbralUlt )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se hara un cierre de los blancos para posteriormente detectar bordes
kernel = np.ones((3,3),np.uint8)
cierre = cv2.morphologyEx(umbralUlt, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Cierre', cierre )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se aplicara erosion para ver si se obtienen mejores contornos
kernel2 = np.ones((12,12),np.uint8)
erosion = cv2.erode(cierre,kernel2,iterations = 1)

# Deteccion de borde
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Se hara una mascara en blanco
cv2.drawContours(ultSoundOriginal, contours, 0, (255, 0, 0), 2)
#cv2.imshow('Regiones', erosion)
#cv2.imshow('Contorno', ultSoundOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(contours[0])

# Recorte de la imagen
isquionTrimm = ultSoundOriginal[76:517, 236:685]
#cv2.imshow('ultSoundTrimm', isquionTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminacion de ruido
dst = cv2.fastNlMeansDenoising(isquionTrimm,None,7,21)
# Aplicacion de umbralizacion
ret,umbralTrimm  = cv2.threshold(dst,1,255,cv2.THRESH_BINARY)
# Apertura
kernel3 = np.ones((7,7),np.uint8)
apertura = cv2.morphologyEx(umbralTrimm, cv2.MORPH_OPEN, kernel3)
# Dilatacion de la imagen
kernel4 = np.ones((25,25),np.uint8)
dilatacion = cv2.dilate(apertura,kernel4,iterations = 1)
cv2.imshow('Sin ruido', dst)
cv2.imshow('Umbral Trimm', umbralTrimm)
cv2.imshow('Apertura', apertura)
cv2.imshow('Diltacion', dilatacion)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Contraste

'''
out = np.zeros_like(ultSoundOriginal) # Extraer el objeto y colocarlo en la imagen de salida
# Cortar imagen
(y, x) = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy+1, topx:bottomx+1]
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
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
