import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

# has four parameters, static_image_mode = false, max_hands = , min_detection_confidence, min_tracking_confidence
# if true, it will do the detection everytimme making the model slow
# we keep it false to track the landmarks instead, if above a certain confidence level
hands = mpHands.Hands()

# this is the function to draw the coordinates on our images
mpDraw = mp.solutions.drawing_utils

PTime = 0
CTime = 0

while True:
    success, frame = cap.read()
    
    # need to convert into rgb as the mediapipe module recognises rgb and not bgr
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # getting the dimensions
    h, w, c = frame.shape

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 8:
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


    CTime = time.time()
    fps = 1 / (CTime - PTime)
    Ptime = CTime
    
    cv2.putText(frame, str((fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    
    cv2.imshow('Feed', frame)
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
cap.release()