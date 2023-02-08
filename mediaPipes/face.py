
import cv2
import mediapipe as mp
import numpy as np


class BlinkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.index_left_eye = [33, 160, 158, 133, 153, 144]
        self.index_right_eye = [362, 385, 387, 263, 373, 380]
        self.EAR_THRESH = 0.26
        self.NUM_FRAMES = 10
        self.aux_counter = 0
        self.blink_counter = 0
        self.cap = cv2.VideoCapture(0)
        """_El método drawing_output es responsable de dibujar los resultados de la detección de 
            parpadeos en la imagen de la cámara. Primero, crea una imagen de numpy con un fondo negro (np.zeros)
            con las mismas dimensiones que la imagen de la cámara. Luego, usa la función cv2.fillPoly para rellenar 
            los contornos de los ojos izquierdo y derecho con un color azul (255, 0, 0). Finalmente, usa la función
            cv2.addWeighted para superponer la imagen anterior con la imagen de la cámara y añade un cuadro y 
        texto para mostrar la cantidad de parpadeos. La imagen resultante es la imagen de salida.
        """

    def drawing_output(self, frame, coordinates_left_eye, coordinates_right_eye):
        aux_image = np.zeros(frame.shape, np.uint8)
        contours1 = np.array([coordinates_left_eye])
        contours2 = np.array([coordinates_right_eye])
        cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
        cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))
        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

        cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)
        cv2.rectangle(output, (202, 0), (265, 50), (255, 0, 0), 2)
        cv2.putText(output, "Num. Parpadeos:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, "{}".format(self.blink_counter), (220, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)

        return output
        """
        El "eye aspect ratio" es una métrica comúnmente utilizada en la detección de seguimiento de 
        ojos en visión por computadora. Se calcula como la relación de las distancias horizontales 
        de los puntos de interés de los ojos y la distancia vertical entre ellos. La idea detrás de 
        esta métrica es que cuando una persona está mirando hacia adelante, estos puntos estarán cerca 
        unos de otros y la relación será alta. En cambio, cuando la persona parpadea o mira hacia arriba 
        o hacia abajo, estos puntos se alejarán y la relación disminuirá. Por lo tanto, la detección de cambios 
        en el eye aspect ratio puede 
        utilizarse para determinar cuándo una persona está mirando hacia otra dirección o parpadeando.
        """

    def eye_aspect_ratio(self, coordinates):
        d_A = np.linalg.norm(
            np.array(coordinates[1]) - np.array(coordinates[5]))
        d_B = np.linalg.norm(
            np.array(coordinates[2]) - np.array(coordinates[4]))
        d_C = np.linalg.norm(
            np.array(coordinates[0]) - np.array(coordinates[3]))

        return (d_A + d_B) / (2 * d_C)

    def start_detection(self):
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1) as face_mesh:

            while True:
                ret, frame = self.cap.read()
                if ret == False:
                    break
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                coordinates_left_eye = []
                coordinates_right_eye = []

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        for index in self.index_left_eye:
                            x = int(face_landmarks.landmark[index].x * width)
                            y = int(face_landmarks.landmark[index].y * height)
                            coordinates_left_eye.append([x, y])
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                            cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                        for index in self.index_right_eye:
                            x = int(face_landmarks.landmark[index].x * width)
                            y = int(face_landmarks.landmark[index].y * height)
                            coordinates_right_eye.append([x, y])
                            cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                    ear_left_eye = self.eye_aspect_ratio(coordinates_left_eye)
                    ear_right_eye = self.eye_aspect_ratio(
                        coordinates_right_eye)
                    ear = (ear_left_eye + ear_right_eye)/2

                    # Ojos cerrados
                    if ear < self.EAR_THRESH:
                        self.aux_counter += 1
                    else:
                        if self.aux_counter >= self.NUM_FRAMES:
                            self.blink_counter += 1
                            self.aux_counter = 0

                    frame = self.drawing_output(
                        frame, coordinates_left_eye, coordinates_right_eye)
                    # if i > 70:
                    # line1 = plotting_ear(pts_ear, line1)

                    # print("pts_ear:", pts_ear)
                cv2.imshow("Frame", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            """
            El método "start_detection" en general es una función utilizada en
            el proceso de detección de rostros en una imagen o video. Este método
            comienza un proceso continuo de detección de rostro, utilizando técnicas 
            de visión por computadora y aprendizaje automático para analizar una secuencia
            de imágenes y determinar la presencia y posición de rostros en ellas.
            
            """


BlinkDetector().start_detection()
