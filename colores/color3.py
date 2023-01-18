import cv2
import numpy as np
cap = cv2.VideoCapture(0)


def dibujarMascaras(mask, color):
    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # version 4.1.0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if (M["m00"] == 0):
                M["m00"] = 1
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

            cv2.putText(frame, '{},{},{}'.format(color, x, y), (x+5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            nuevoContorno = cv2.convexHull(c)

            cv2.drawContours(frame, [nuevoContorno], 0, (0, 0, 255), 3)


AzulBajo = np.array([100, 100, 20], np.uint8)
AzulAlto = np.array([125, 255, 255], np.uint8)
rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([8, 255, 255], np.uint8)
rojoBajo2 = np.array([175, 100, 20], np.uint8)
rojoAlto2 = np.array([179, 255, 255], np.uint8)
VerdeBajo = np.array([35, 100, 20], np.uint8)
VerdeAlto = np.array([70, 255, 255], np.uint8)

while cap.isOpened():
    _, frame = cap.read()

    # read the camera frame

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    MascaraAzul = cv2.inRange(frameHSV, AzulBajo, AzulAlto)
    MascaraVerde = cv2.inRange(frameHSV, VerdeBajo, VerdeAlto)
    MascaraRoja1 = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
    MascaraRoja2 = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
    MascaraRoja = cv2.add(MascaraRoja1, MascaraRoja2)
    dibujarMascaras(MascaraAzul, 'Azul')
    dibujarMascaras(MascaraVerde, 'Verde')
    dibujarMascaras(MascaraRoja, 'Rojo')

    # show the frame
    cv2.imshow('frame', frame)
    #cv2.imshow('MascaraRoja', MascaraSum)
    cv2.waitKey(1)
    # wait for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
