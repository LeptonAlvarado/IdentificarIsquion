import cv2
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as m
import scipy.signal

# Se crea una funcion para saber caracteristicas de la matriz
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

########################### Obtaining the image ##########################
# Si la imagen esta en la misma carpeta que el programa se puede poner solo el nombre de la imagen
# Si no, se necesita poner la direccion
# cv2.imread('imagen',bandera)
# bandera = 0 Escala de grises, = 1 A Color, = -1 Carga la imagen como tal, incluyendo el canal alfa
ultSoundOriginal = cv2.imread('C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/IT SandraITXXXX0E_Frame10.jpg', 0)

# Se aplica un umbral en el que si es diferente de 0 se haga 1
# https://www.pyimagesearch.com/2014/09/08/thresholding-simple-image-segmentation-using-opencv/ 
ret,umbralUlt  = cv2.threshold(ultSoundOriginal,0,255,cv2.THRESH_BINARY)
#cv2.imshow('Umbral', umbralUlt )
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Se hara un cierre de los blancos para posteriormente detectar bordes
kernel = np.ones((3,3),np.uint8)
ultCierre = cv2.morphologyEx(umbralUlt, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Cierre', ultCierre )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se aplicara erosion para ver si se obtienen mejores contornos
kernel2 = np.ones((12,12),np.uint8)
ultErosion = cv2.erode(ultCierre,kernel2,iterations = 1)
cv2.imshow('Cierre', ultErosion )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Deteccion de bordes
contoursUlt, hierarchyUlt = cv2.findContours(ultErosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
# Find object with the biggest bounding box
mx = (0,0,0,0)      # biggest bounding box so far
mx_area = 0
for cont in contoursUlt:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h
    if area > mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx

# Recorte del Isquion
isquionTrimm=ultSoundOriginal[y:y+h,x:x+w]
cv2.imshow('ultSoundTrimm', isquionTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

denoisingIsquion = cv2.fastNlMeansDenoising((isquionTrimm),None,7,21)
cv2.imshow('Eliminacion Sin Ruido', denoisingIsquion)
cv2.waitKey(0)
cv2.destroyAllWindows()

normalIsqTrimm = cv2.normalize(denoisingIsquion.astype(np.float32),0,255,cv2.NORM_HAMMING)
#valorImagen(normalIsqTrimm)
cv2.imshow('Normalizado', normalIsqTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

equ = cv2.equalizeHist(normalIsqTrimm.astype(np.uint8))
cv2.imshow('Chale', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Eliminacion de ruido
#isquionDifuminado = cv2.fastNlMeansDenoising(isquionTrimm,None,7,21)
# Aplicacion de umbralizacion
ret,umbralTrimm  = cv2.threshold(equ,1,255,cv2.THRESH_BINARY)
cv2.imshow('Umbral Trimm', umbralTrimm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Deteccion de bordes
contoursIsq, hierarchyIsq = cv2.findContours(umbralTrimm.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mx1 = (0,0,0,0)      # biggest bounding box so far
mx_area1 = 0
for cont1 in contoursIsq:
    x1,y1,w1,h1 = cv2.boundingRect(cont1)
    area1 = w1*h1
    if area1 > mx_area1:
        mx1 = x1,y1,w1,h1
        mx_area1 = area1
x1,y1,w1,h1 = mx1

isquionBone=isquionTrimm[y1:y1+h1,x1:x1+w1]
cv2.imshow('ultSoundTrimm', isquionBone)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
size = np.size(cierreNormal)
skel = np.zeros(cierreNormal.shape,np.uint8)

#cv2.MORPH_CROSS,(3,3) No cambiar a valores menores a 3
# Ezqueletizacion http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
while( not done):
    eroded = cv2.erode(cierreNormal,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(cierreNormal,temp)
    skel = cv2.bitwise_or(skel,temp)
    cierreNormal = eroded.copy()
 
    zeros = size - cv2.countNonZero(cierreNormal)
    if (zeros==size):
        done = True

cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
# Detectar Lineas
copySkel = cv2.ctvColor(skel,cv2.CV2_GRAY2BGR)
minLineLength = 100
maxLineGap = 1
lines = cv2.HoughLinesP(copySkel,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(skel,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('detected lines',skel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detectar Circulos
copySkel = skel
circles = cv2.HoughCircles(skel,cv2.HOUGH_GRADIENT,1,30,
                            param1=80,param2=20,minRadius=1,maxRadius=40)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(copySkel,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(copySkel,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',copySkel)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''