import cv2
import numpy as np
import HandTracking_GestureRecognition_Module as hgm

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    canvas = np.zeros((720, 1280, 3), dtype = 'uint8')

    xp, yp = 0, 0

    detector = hgm.HandDetector()

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.FindHands(frame, True)
        lm_list = detector.FindPositions(frame, 0)

        if len(lm_list):
            fingers = detector.FindGesture()
            xi, yi = lm_list[8][1:]

            if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                cv2.circle(frame, (xi, yi), 20, (0, 0, 255), -1)
                if xp == 0 and yp == 0:
                    xp, yp = xi, yi
                
                cv2.line(canvas, (xp, yp), (xi, yi), (0, 0, 255), 20)
                xp, yp = xi, yi
            
            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                xp, yp = 0, 0

            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                cv2.circle(frame, (xi, yi), 60, (0, 0, 0), -1)
                cv2.circle(canvas, (xi, yi), 60, (0, 0, 0), -1)
                xp, yp = 0, 0

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, ImgInv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        ImgInv = cv2.cvtColor(ImgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, ImgInv)
        frame = cv2.bitwise_or(frame, canvas)

    
        cv2.imshow('Live', frame)

        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()