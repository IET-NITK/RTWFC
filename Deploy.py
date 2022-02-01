from Hand_Tracking import Coordinates
import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = Coordinates(frame)
    
    cv2.imshow('Feed', frame)
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
cap.release()