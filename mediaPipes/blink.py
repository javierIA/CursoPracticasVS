import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)


parpadeo = False  # variable para saber si se esta parpadeando
cortador= 0  # variable para contar los blink 


mpDibujo=mp.solutions.drawing_utils
ConfigDibujo=mpDibujo.DrawingSpec(thickness=1, circle_radius=1) # configuracion para dibujar

mpCaraMalla=mp.solutions.face_mesh
ConfigCaraMalla=mpCaraMalla.FaceMesh(max_num_faces=1) # configuracion para la cara
px=[]
py=[]
lista=[]

while(True):
 _,frame= cap.read()
 frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # convertir a RGB
 resultados=ConfigCaraMalla.process(frameRGB) 
 if resultados.multi_face_landmarks:
   for cara in resultados.multi_face_landmarks:
    #mpDibujo.draw_landmarks(frame,cara,ConfigCaraMalla.FACE_CONNECTIONS,ConfigDibujo,ConfigDibujo)
    
    # obtener los puntos de la cara
    
    for ids,puntos in enumerate(cara.landmark):
     alto,ancho,_=frame.shape
     x,y=int(puntos.x*ancho),int(puntos.y*alto)
     px.append(x)
     py.append(y)
     lista.append([ids,x,y])
     if len(lista)>500:
       print(lista)