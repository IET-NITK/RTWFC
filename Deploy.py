from Hand_Tracking import Coordinates
import numpy as np
import cv2

# When the mouse moves fast, you cannot draw fast enough to keep up with the mouse events. To confirm this, modify your code to capture the mouse coordinates without drawing and add them in a list. Add a keypress handler to draw the captured points. If this is the case, you could draw a circle on a small transparent image once. You can then overlay that small image instead of drawing a circle which involves too many calculations in the space of a fraction of a second. Give this a shot and advise.

blank = np.zeros((480, 640, 3), dtype = 'uint8')
blank.fill(255)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame, x, y = Coordinates(frame, 0, 0)
    
    cv2.imshow('Feed', frame)
    cv2.imshow('Paint', blank)

    cv2.circle(blank, (x, y), 25, (0, 0, 255), cv2.FILLED)

    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
cap.release()