import cv2
import numpy as np 
from pyzbar.pyzbar import decode


cap = cv2.VideoCapture(0)
# capture from camera at location 0
while cap.isOpened():
    _, frame = cap.read()
    # read the camera frame
    cv2.imshow('frame', frame)
    # show the frame
    for barcode in decode(frame):
        decoded_data= barcode.data.decode('utf-8')
        rect = barcode.rect
        
        
        pts= np.array([barcode.polygon], np.int32)
        cv2.polylines(frame,[pts],True,(255,0,255),3)
        cv2.putText(frame,str(decoded_data),(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        
    cv2.imshow('frame', frame)
    # wait for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
