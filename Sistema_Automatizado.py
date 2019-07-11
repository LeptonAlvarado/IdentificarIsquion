import glob           # Libreria para leer los archivos
import cv2            # Libreria para hacer el procesamiento de imagenes
import numpy as np    # Libreria para trabajar con las matrices de las imagenes

# Esta variable guarda la direccion del archivo
# Se tomo hasta el nombre de la imagen menos los ultimos numeros
my_path = 'C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/IT SandraITXXXX0E_Frame'

# Esta variable almacena la funcion glob la cual buscara todos los archivos con terminacion .jpg
files = glob.glob(my_path + '*.jpg')

# Aqui se leeran todos los archivos para poder trabajar con ellos y finalmente guardarlos
# Esta funcion lee las imagenes de forma 1,10,100,101,102,...,109,11,110,111,...
for file in files:
    # Si todo esta bien entrara a esta parte del codigo
    try:
        i=0
        i+=1
        # En esta parte se lee la imagen
        ultraSonido = cv2.imread(file,0)
        # Se recorta la imagen, para que solo quede la parte donde se encuentra el ultrasonido
        ultSonTrimm = ultraSonido[76:517, 236:685] 
        # De esa parte se recorta para dejar solo la parte del hueso
        isquionTrimm = ultSonTrimm[0:168, 0:449]
        cv2.imshow('aver',isquionTrimm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #print(files & '/n')
        # Al final se debe crear otra carpeta con los nombres del archivo
    # En caso de que encuentre un problema entrera a esta parte    
    except:
        print('Cant import ' + file)