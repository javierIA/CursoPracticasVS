import cv2
import numpy as np
cap = cv2.VideoCapture(0)

rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([8, 255, 255], np.uint8)

rojoBajo2 = np.array([175, 100, 20], np.uint8)
rojoAlto2 = np.array([179, 255, 255], np.uint8)

while cap.isOpened():
    _, frame = cap.read()

    # read the camera frame

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    MascaraUno = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
    MascaraDos = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
    MascaraRoja = cv2.add(MascaraUno, MascaraDos)
    MascaraVis = cv2.bitwise_and(frame, frame, mask=MascaraRoja)
    # show the frame
    cv2.imshow('frame', frame)
    cv2.imshow('MascaraRoja', MascaraRoja)
    cv2.imshow('MascaraVis', MascaraVis)
    cv2.waitKey(1)
    # wait for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
