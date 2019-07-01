import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Se crea una funcion pra saber cual es el valor mas alto de la matriz
def valorMaxImagen (imagen)
    maxPixel = np.amax(imagen)
    return maxPixel


# Con este codigo se muestra la imagen tal cual esta
ultSoundOriginal = cv2.imread('prueba.jpg', 1)
'''
La imagen tiene informacion que no nos interesa,
por lo que se recortara la parte que nos interesa 
Primero veremos el tamaño de la imagen
Se le da click derecho a la imagen sin abrir y buscamos en propiedades
En la seccion de detalles estara la medida en pixeles
Con matplotlib.pyplot se crea una grafica para saber donde estan los puntos que nos interesan
Para esto se necesita mover el cursor  en los puntos de la imagen que deseamos
'''
# Muestra la imagen en su grafica
ax = plt.subplot2grid ((600, 800),(0,0), rowspan= 600, colspan=800)
ax.imshow(ultSoundOriginal)
plt.show()
# Recorte de imagen = img[y, x]
ultSoundTrimm = ultSoundOriginal[63:523, 229:690]
cv2.imshow('ultSoundTrimm', ultSoundTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Histograma
hist = cv2.calcHist([ultSoundTrimm], [0], None, [256], [0, 256])
plt.plot(hist, color='gray' )

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()

# Convertir a grises por si las dudas
ultSoundGray = cv2.cvtColor(ultSoundTrimm,cv2.COLOR_BGR2GRAY)

# Se obtiene el valor maximo de la matriz


















'''
# Normalizacion de los datos
norm1 = cv2.normalize(ultSoundGray, 0, 255, cv2.NORM_MINMAX)
norm2 = cv2.normalize(ultSoundGray, 0, 255, cv2.NORM_L1)
norm3 = cv2.normalize(ultSoundGray, 0, 255, cv2.NORM_L2)
norm4 = cv2.normalize(ultSoundGray, 0, 255, cv2.NORM_HAMMING)

cv2.imshow('Normalizacion 1', norm1)
cv2.imshow('Normalizacion 2', norm2)
cv2.imshow('Normalizacion 3', norm3)
cv2.imshow('Normalizacion 4', norm4)

cv2.waitKey(-1)
cv2.destroyAllWindows()


# Gradiante
sobelx8u = cv2.Sobel(ultSoundTrimm,cv2.CV_8U,1,0,ksize=5)
#Utilizando cv2.CV_64F. Luego toma el valor absoluto y hace la conversión a cv2.CV_8U
sobelx64f = cv2.Sobel(ultSoundTrimm,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(ultSoundTrimm,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()


# Deteccion de bordes
ultSoundGray = cv2.cvtColor(sobel_8u,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(ultSoundGray,127,255,cv2.THRESH_BINARY)
cv2.imshow('xd', thresh1)
cv2.waitKey(0)
contornos, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sobel_8u, contornos, -1, (0,255,0), 3)
cv2.imshow('ultSoundContornos', sobel_8u)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''







