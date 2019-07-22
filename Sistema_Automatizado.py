import glob           # Libreria para leer los archivos
import cv2            # Libreria para hacer el procesamiento de imagenes
import numpy as np    # Libreria para trabajar con las matrices de las imagenes
import os             # Libreria para leer nombre de archivos

def nombreImagen (path_carpeta):
    listaDeNombresImg = []
    items = os.listdir(path_carpeta)
    for imagen in items:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            nombre = imagen
            # Dentro del parentesis se pone todo lo que esta antes de los numeros del nombre de la imagen
            separacion = nombre.split('IT SandraITXXXX0E_Frame')
            # Se pone int al comienzo para convertirlo a una variable de punto entero
            # En la variable separacionse le pone [1] ya que el vaalor [0] e s todo lo que esta antes del numero
            # Ahora en la variable numero se guarda el split de separacion pero esta ves el [0] que contiene el numero
            # Ya que el [1] contendria la direccion del archivo
            numero = int(separacion[1].split('.jpg')[0])
            listaDeNombresImg.append(numero)
        # En caso de que encuentre un problema entrera a esta parte 
        except Exception as error:
            print(error)
    
    # Regresa solo los numeros del nombre de la imagen en una lista de enteros
    return (listaDeNombresImg)

def obtenerImagen(path_carpeta):
    listaUltrasonido = []
    # Esta variable almacena la funcion glob la cual buscara todos los archivos con terminacion .jpg
    files = glob.glob(path_carpeta + '*.jpg')

    # Aqui se leeran todos los archivos para poder trabajar con ellos y finalmente guardarlos
    # Esta funcion lee las imagenes de forma 1,10,100,101,102,...,109,11,110,111,...
    for file in files:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            # En esta parte se lee la imagen
            ultraSonido = cv2.imread(file,0)
            # Se aplica un umbral
            ret,umbralizacionUltrasonido  = cv2.threshold(ultraSonido,0,255,cv2.THRESH_BINARY)
            
            # Se hara un cierre a la imagen
            kernel = np.ones((3,3),np.uint8)
            cierreUltrasonido = cv2.morphologyEx(umbralizacionUltrasonido, cv2.MORPH_CLOSE, kernel)

            # Se aplicara erosion obtener mejor contorno en el ultrasonido
            kernel2 = np.ones((12,12),np.uint8)
            erosionUltrasonido = cv2.erode(cierreUltrasonido,kernel2,iterations = 1)

            # Deteccion de bordes
            contornosUltrasonido, jerarquiaUltrasonido = cv2.findContours(erosionUltrasonido, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
            # Encuentra el contorno con mayor area
            mx = (0,0,0,0)
            mx_area = 0
            for cont in contornosUltrasonido:
                x,y,w,h = cv2.boundingRect(cont)
                area = w*h
                if area > mx_area:
                    mx = x,y,w,h
                    mx_area = area
            x,y,w,h = mx

            # Recorte del Isquion
            isquionTrimm=ultraSonido[y:y+h,x:x+w]
            cv2.imshow('ultSoundTrimm', isquionTrimm)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            listaUltrasonido.append(isquionTrimm)
            # Al final se debe crear otra carpeta con los nombres del archivo
        # En caso de que encuentre un problema entrera a esta parte 
        except Exception as error:
            print(error)
    return(listaUltrasonido)

def obtenerIsquion(ultrasonidos = []):
    listaIsquion = []
    for ultrasonido in ultrasonidos:
        try:
            denoisingIsquion = cv2.fastNlMeansDenoising((ultrasonido),None,7,21)
            normalizacionIsquion = cv2.normalize(denoisingIsquion.astype(np.float32),0,255,cv2.NORM_HAMMING)
            ecualizacionIsquion = cv2.equalizeHist(normalizacionIsquion.astype(np.uint8))
            ret,umbralIsquion  = cv2.threshold(ecualizacionIsquion,1,255,cv2.THRESH_BINARY)
            contoursIsq, hierarchyIsq = cv2.findContours(umbralIsquion.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mx1 = (0,0,0,0)      # biggest bounding box so far
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

            isquionBone=ultrasonido[y1:y1+h1,x1:x1+w1]
            cv2.imshow('ultSoundTrimm', isquionBone)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as error:
            print(error)

# Esta variable guarda la direccion de los archivos
my_path = 'C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/'

nomImgs = nombreImagen(my_path)
cutImgs = obtenerImagen(my_path) 
obtenerIsquion(cutImgs)
