# Imports
import numpy as np
from cv2 import cv2
import os

# Captura a Camera
cap = cv2.VideoCapture(0)

# carrega o haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# LOAD RECOGNIZER pip install opencv_contrib_python
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Loop
while(True):
    
    #lê o frame

    ret, frame = cap.read()

    # frame cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detecta a face.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # junta os frames.
    for (x,y,w,h) in faces:    

        # recorta o rosto
        roi = frame[y:y+h, x:x+w]

        roi = cv2.resize(roi, (5,5))
        roi = cv2.resize(roi, (w,h))

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        frame[y:y+h, x:x+w] = roi
    # mostra o frame na tela // print
    cv2.imshow('frame', frame)

    # WAITKEY
    key = cv2.waitKey(15)

    # quebra o loop se o Q for apertado.
    if key & 0xFF == ord('q'):
        break
        
cap.release()

# Destroi todas as janelas.
cv2.destroyAllWindows()
