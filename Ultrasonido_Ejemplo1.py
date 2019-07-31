import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Se crea una funcion para saber caracteristicas de la matriz de la imagen
def valorImagen (imagen):
    maxPixel = np.max(imagen)
    minPixel = np.min(imagen)
    medianPixel = np.median(imagen)
    meanPixel = np.mean(imagen)
    print(minPixel)
    print(maxPixel)
    print(medianPixel)
    print(meanPixel)
    print(imagen)

########################### Obtaining image ##########################
# Si la imagen esta en la misma carpeta que el programa se puede poner solo el nombre de la imagen
# Si no, se necesita poner la direccion
# cv2.imread('imagen',bandera)
# bandera = 0 Escala de grises, = 1 A Color, = -1 Carga la imagen como tal, incluyendo el canal alfa
imagenOriginal = cv2.imread('C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/IT SandraITXXXX0E_Frame90.jpg', 0)


########################## Detecting ultrasound ##############################
# Se aplica un umbral en el que si es diferente de 0 (negro) se haga 255 (blanco)
# https://www.pyimagesearch.com/2014/09/08/thresholding-simple-image-segmentation-using-opencv/ 
ret,umbralImagen  = cv2.threshold(imagenOriginal,0,255,cv2.THRESH_BINARY)
cv2.imshow('Umbral', umbralImagen )
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Se le aplicara un cierre a la imagen
kernel = np.ones((3,3),np.uint8)
cierreImagen = cv2.morphologyEx(umbralImagen, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Cierre', cierreImagen )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se aplicara erosion para ver si se obtienen mejores contornos
kernel2 = np.ones((12,12),np.uint8)
erosionImagen = cv2.erode(cierreImagen,kernel2,iterations = 1)
cv2.imshow('Cierre', erosionImagen )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Deteccion de bordes
contoursImg, hierarchyImg = cv2.findContours(erosionImagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
# Detecta el area mas grande
mx = (0,0,0,0)      
mx_area = 0
for cont in contoursImg:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h
    if area > mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx


############################# Cut Ultrasound ##################################
# Recorte del Isquion
ultrasoundTrimm=imagenOriginal[y:y+h,x:x+w]
cv2.imshow('ultSoundTrimm', ultrasoundTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################### Detecting Isquion #################################
# Eliminacion del ruido
denoisingUltsound = cv2.fastNlMeansDenoising((ultrasoundTrimm),None,7,21)
cv2.imshow('Eliminacion Sin Ruido', denoisingUltsound)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalizacion del Ultrasonido
normalUltsound = cv2.normalize(denoisingUltsound.astype(np.float32),0,255,cv2.NORM_HAMMING)
#valorImagen(normalUltsound)
cv2.imshow('Normalizado', normalUltsound)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Equalizacion del Ultrasonido
equalizacionUtlsound = cv2.equalizeHist(normalUltsound.astype(np.uint8))
cv2.imshow('Ecualizacion', equalizacionUtlsound)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aplicacion de umbralizacion en Ultrasonido
ret,umbralTrimm  = cv2.threshold(equalizacionUtlsound,1,255,cv2.THRESH_BINARY)
cv2.imshow('Umbral Trimm', umbralTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Deteccion de bordes
contoursIsq, hierarchyIsq = cv2.findContours(umbralTrimm.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Detecta el area mas garnde
mx1 = (0,0,0,0)      
mx_area1 = 0
for cont1 in contoursIsq:
    x1,y1,w1,h1 = cv2.boundingRect(cont1)
    area1 = w1*h1
    if area1 > mx_area1:
        mx1 = x1,y1,w1,h1
        mx_area1 = area1
x1,y1,w1,h1 = mx1
x1 -= 25
y1 -= 10
w1 += 115
h1 += 100

############################# Cut Isquion #################################
isquionBone=ultrasoundTrimm[y1:y1+h1,x1:x1+w1]
cv2.imshow('ultSoundTrimm', isquionBone)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################ Processing Isquion ##############################
# Normalizacion del Isquion
isquionNormalizado = cv2.normalize(isquionBone.astype(np.float32),0,255,cv2.NORM_HAMMING)
cv2.imshow('Isquion Normalizado', isquionNormalizado)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Umbralizacion del Isquion
ret, isquionUmbral = cv2.threshold(isquionNormalizado,1,255,cv2.THRESH_BINARY)
cv2.imshow('Isquion Umbralizado', isquionUmbral)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cierre del Isquion
kernel3 = np.ones((8,8),np.uint8)
isquionCierre = cv2.morphologyEx(isquionUmbral, cv2.MORPH_CLOSE, kernel3)
cv2.imshow('Cierre', isquionCierre )
cv2.waitKey(0)
cv2.destroyAllWindows()

size = np.size(isquionCierre)
skel = np.zeros(isquionCierre.shape,np.uint8)

#cv2.MORPH_CROSS,(3,3) No cambiar a valores menores a 3
# Ezqueletizacion del Isquion http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
while( not done):
    eroded = cv2.erode(isquionCierre.astype(np.uint8),element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(isquionCierre.astype(np.uint8),temp)
    skel = cv2.bitwise_or(skel,temp)
    isquionCierre = eroded.copy()
 
    zeros = size - cv2.countNonZero(isquionCierre.astype(np.uint8))
    if (zeros==size):
        done = True

cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################### Deteccion de Caracteristicas ##############################
'''
Esta parte ya no se logro hacer pero se pensaba una deteccion de circulos de Hough
img = isquionBone
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''