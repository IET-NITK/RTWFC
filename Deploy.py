import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    
    cv2.imshow('Feed', frame)
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
cap.release()