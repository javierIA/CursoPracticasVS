import cv2 
cap = cv2.VideoCapture(0)
# capture from camera at location 0
while cap.isOpened():
    _, frame = cap.read()
    # read the camera frame
    cv2.imshow('frame', frame)
    # show the frame
    cv2.waitKey(1)
    # wait for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
