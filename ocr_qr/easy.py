import easyocr
import cv2 
from pathlib import Path
reader = easyocr.Reader(['es']) # this needs to run only once to load the model into memory

path= Path('ocr_qr\images\ocr.jpg')


img= cv2.imread(str(path))
img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
#cambiar color de la imagen a escala de grises
rango= 100
#_,img= cv2.threshold(img,rango,255,cv2.THRESH_BINARY_INV)
#usar en casos que la imagen no se vea bien
result = reader.readtext(img, detail = 0)
for i in result:
    print(i)
cv2.imshow('Imagen',img)
cv2.waitKey(0)

