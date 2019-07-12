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

def obtenerImagen(path_carpeta)
    listaImagenes = []
    # Esta variable almacena la funcion glob la cual buscara todos los archivos con terminacion .jpg
    files = glob.glob(path_carpeta + '*.jpg')

    # Aqui se leeran todos los archivos para poder trabajar con ellos y finalmente guardarlos
    # Esta funcion lee las imagenes de forma 1,10,100,101,102,...,109,11,110,111,...
    for file in files:
        # Si todo esta bien entrara a esta parte del codigo
        try:
            # En esta parte se lee la imagen
            ultraSonido = cv2.imread(file,0)
            # Se recorta la imagen, para que solo quede la parte donde se encuentra el ultrasonido
            ultSonTrimm = ultraSonido[76:517, 236:685] 
            # De esa parte se recorta para dejar solo la parte del hueso
            isquionTrimm = ultSonTrimm[0:168, 0:449]
            cv2.imshow('aver',isquionTrimm)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            listaImagenes.append(isquionTrimm)
            # Al final se debe crear otra carpeta con los nombres del archivo
        # En caso de que encuentre un problema entrera a esta parte 
        except Exception as error:
            print(error)


# Esta variable guarda la direccion de los archivos
my_path = 'C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/'

nomImgs = nombreImagen(my_path)
cutImgs = obtenerImagen(my_path)
