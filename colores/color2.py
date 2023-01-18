import cv2
import numpy as np
cap = cv2.VideoCapture(0)

AzulBajo = np.array([100, 100, 20], np.uint8)
AzulAlto = np.array([125, 255, 255], np.uint8)


while cap.isOpened():
    _, frame = cap.read()

    # read the camera frame

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    MascaraUno = cv2.inRange(frameHSV, AzulBajo, AzulAlto)
    # _,contorno,_ = cv2.findContours(MascaraUno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) version 3.4.2
    contornos, _ = cv2.findContours(
        MascaraUno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # version 4.1.0

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if (M["m00"] == 0):
                M["m00"] = 1
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, 'Centro{},{}'.format(x, y), (x+5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            nuevoContorno = cv2.convexHull(c)

            cv2.drawContours(frame, [nuevoContorno], 0, (0, 0, 255), 3)

    #cv2.drawContours(frame, contornos, -1, (0, 255, 0), 3)
    #MascaraSum = cv2.add(MascaraUno, MascaraDos)
    MascaraVis = cv2.bitwise_and(frame, frame, mask=MascaraUno)
    # show the frame
    cv2.imshow('frame', frame)
    #cv2.imshow('MascaraRoja', MascaraSum)
    cv2.imshow('MascaraVis', MascaraVis)
    cv2.waitKey(1)
    # wait for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
