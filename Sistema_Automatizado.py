import glob
import cv2

cv_img = []
my_path = 'C:/Users/josue/OneDrive/Escritorio/Ultrasonido/Dummi Right/Series_1/IT SandraITXXXX0E_Frame'

files = glob.glob(my_path + '*.jpg')

for file in files:
    try:
        n = cv2.imread(file)
        cv2.imshow('aver',n)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Al final se debe crear otra carpeta con los nombres del archivo
    except:
        print('Cant import ' + file)