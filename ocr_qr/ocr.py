import cv2 
import numpy 
from pathlib import Path 
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'



# cuando no detecte images usar la ruta como esta 
# C:\Users\Usuario\Documents\GitHub\ocr_qr\images\ocr.jpg
path= Path('ocr_qr\images\ocr.jpg')
custom_oem_psm_config = r'--oem 1 --psm 3'
#los valores de oem y psm se pueden cambiar para mejorar la deteccion
#ver tessaract docs para mas informacion

img= cv2.imread(str(path))
img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
#cambiar color de la imagen a escala de grises
rango= 100
#_,img= cv2.threshold(img,rango,255,cv2.THRESH_BINARY_INV)
#usar en casos que la imagen no se vea bien

cv2.imshow('Imagen',img)
cv2.waitKey(0)
print(pytesseract.image_to_string(img,lang='spa',config=custom_oem_psm_config))

cv2.imshow('Imagen',img)
cv2.waitKey(0)
