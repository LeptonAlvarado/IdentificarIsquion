import glob           # Libreria para leer los archivos
import cv2            # Libreria para hacer el procesamiento de imagenes
import numpy as np    # Libreria para trabajar con las matrices de las imagenes
import os             # Libreria para leer nombre de archivos

'''
Esta funcion recibe el path de la carpeta donde se encuentran las imagenes.
Asi como tambien el nombre general de la imagen sin el numero.
Y regresa una lista con los numeros de la imagen
'''
def numeroImagen (path_carpeta, nombreGeneralImagen):
    listaDeNombresImg = []
    items = os.listdir(path_carpeta)
    for imagen in items:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            nombre = imagen
            # Dentro del parentesis se pone todo lo que esta antes de los numeros del nombre de la imagen
            separacion = nombre.split(nombreGeneralImagen)
            # Se pone int al comienzo para convertirlo a una variable de punto entero
            # En la variable separacionse le pone [1] ya que el valor [0] e s todo lo que esta antes del numero
            # Ahora en la variable numero se guarda el split de separacion pero esta ves el [0] que contiene el numero
            # Ya que el [1] contendria la direccion del archivo
            numero = int(separacion[1].split('.jpg')[0])
            # El numero obtenido se agrega a una lista
            listaDeNombresImg.append(numero)
        # En caso de que encuentre un problema entrera a esta parte 
        except Exception as error:
            print(error)
    
    # Regresa solo los numeros del nombre de la imagen en una lista de enteros
    return (listaDeNombresImg)

'''
Esta funcion recibe el path de la carpeta donde se encuentran las imagenes.
Y detecta el ultrasonido dependiendo el area 
Y devuelve solo el ultrasonido
'''
def obtenerImagen (path_carpeta):
    listaUltrasonido = []
    # Esta variable almacena la funcion glob la cual buscara todos los archivos con terminacion .jpg
    files = glob.glob(path_carpeta + '*.jpg')

    # Aqui se leeran todos los archivos para poder trabajar con ellos y finalmente guardarlos
    # Esta funcion lee las imagenes de forma 1,10,100,101,102,...,109,11,110,111,...
    for file in files:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            # En esta parte se lee la imagen
            imagenUltrasonido = cv2.imread(file,0)
            # Se aplica un umbral
            ret,umbralizacionUltrasonido  = cv2.threshold(imagenUltrasonido,0,255,cv2.THRESH_BINARY)
            
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

            # Recorte del Ultrasonido
            ultrasonido=imagenUltrasonido[y:y+h,x:x+w]
            cv2.imshow('ultSoundTrimm', ultrasonido)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            # Se agrega el ultrasonido recortado a una lista
            listaUltrasonido.append(ultrasonido)
            # Al final se debe crear otra carpeta con los nombres del archivo
        # En caso de que encuentre un problema entrera a esta parte 
        except Exception as error:
            print(error)
    return(listaUltrasonido)

'''
Esta funcion recibe la lista de ultrasonido
Y se detecta el hueso dependiendo el area
Y devuelve solo el hueso detectado
'''
def obtenerIsquion (ultrasonidos = []):
    listaIsquion = []
    # En este for se leen las imagenes de la lista
    for ultrasonido in ultrasonidos:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            # Eliminacion del ruido del ultrasonido
            denoisingUltsound = cv2.fastNlMeansDenoising((ultrasonido),None,7,21)
            # Normalizacion del ultrasonido
            normalizacionUltsound = cv2.normalize(denoisingUltsound.astype(np.float32),0,255,cv2.NORM_HAMMING)
            # Ecualizacion del ultrasonido
            ecualizacionUltsound = cv2.equalizeHist(normalizacionUltsound.astype(np.uint8))
            # Umbralizacion del ultrasonido
            ret,umbralUltsound  = cv2.threshold(ecualizacionUltsound,1,255,cv2.THRESH_BINARY)
            contoursUlt, hierarchyUlt = cv2.findContours(umbralUltsound.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mx1 = (0,0,0,0)      # biggest bounding box so far
            mx_area1 = 0
            for conts in contoursUlt:
                x1,y1,w1,h1 = cv2.boundingRect(conts)
                area1 = w1*h1
                if area1 > mx_area1:
                    mx1 = x1,y1,w1,h1
                    mx_area1 = area1
            x1,y1,w1,h1 = mx1
            x1 -= 30
            y1 -= 12
            w1 += 115
            h1 += 100
            
            # Recorte del isquion
            isquionBone=ultrasonido[y1:y1+h1,x1:x1+w1]
            listaIsquion.append(isquionBone)
            cv2.imshow('ultSoundTrimm', isquionBone)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as error:
            print(error)
    return(listaIsquion)

'''
Esta funcion recibe la lista del numero de imagenes
ASi como la lista de las imagenes recortadas que se quieren guardar
El path destino donde se guardaran las imagenes
El nombre general con el que se guardaran las imagenes 
'''
def guardarImagenes (numeroImagenes =[], imagenes=[], pathDestino,  nombreGeneral):
    for imagen in imagenes:
        try:
            contador = 0
            cv2.imwrite(os.path.join(pathDestino, nombreGeneral + numeroImagen[contador]+'.jpg'),imagen)
            cv2.waitKey(0)
            contador += 1
        except Exception as error:
            print(error)


# Esta variable guarda la direccion de los archivos
my_path = 'C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Left/Series_2/'
nombreConstanteImagen = 'IT SandraITXXXX0I_Frame'

numImgs = nombreImagen(my_path, nombreConstanteImagen)
cutImgs = obtenerImagen(my_path) 
cutIsquion = obtenerIsquion(cutImgs)
