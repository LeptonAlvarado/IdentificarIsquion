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
'''
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

ruidoTrimm = ultSoundOriginal[260:523, 229:690]
cv2.imshow('ultSoundTrimm', ruidoTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist = cv2.calcHist([ruidoTrimm], [0], None, [256], [0, 256])
plt.plot(hist, color='gray' )

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()

isquionTrimm = ultSoundOriginal[128:250, 361:595]
cv2.imshow('ultSoundTrimm', isquionTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist = cv2.calcHist([isquionTrimm], [0], None, [256], [0, 256])
plt.plot(hist, color='gray' )

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()
'''
negroTrimm = ultSoundOriginal[300:400, 100:200]
cv2.imshow('ultSoundTrimm', negroTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''

gaussiana = cv2.GaussianBlur(ultSoundTrimm, (5,5), 0)
cv2.imshow('cl1', gaussiana)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
cl1 = clahe.apply(ultSoundTrimm)

result = cv2.addWeighted(cl1,2,np.zeros(cl1.shape, cl1.dtype),0,50)
cv2.imshow('cl1', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
equ = cv2.equalizeHist(ultSoundTrimm)
res = np.hstack((ultSoundTrimm,equ))
cv2.imshow('xdxd',res)
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imshow('xdxd',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist = cv2.calcHist([cl1], [0], None, [256], [0, 256])
plt.plot(hist, color='gray' )

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()


dst = cv2.fastNlMeansDenoising(cl1,None,10,7,21)
cv2.imshow('xdxd',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# Se obtiene el valor maximo de la matriz
maxValuePixel = valorMaxImagen(negroTrimm)
#ultSoundBW = balanceDeBlancos(ultSoundGray, maxValuePixel)





'''

# Gradiante
sobelx8u = cv2.Sobel(dst,cv2.CV_8U,1,0,ksize=5)
#Utilizando cv2.CV_64F. Luego toma el valor absoluto y hace la conversión a cv2.CV_8U
sobelx64f = cv2.Sobel(dst,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(dst,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

# Deteccion de bordes
ultSoundGray = dst
ret,thresh1 = cv2.threshold(ultSoundGray,127,255,cv2.THRESH_BINARY)
cv2.imshow('xd', thresh1)
cv2.waitKey(0)
contornos, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dst, contornos, -1, (0,255,0), 3)
cv2.imshow('ultSoundContornos', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''







