import cv2
import mediapipe as mp

# compulsory to implement mediapipe functions
mpHands = mp.solutions.hands

# has four parameters, static_image_mode = false, max_hands, min_detection_confidence, min_tracking_confidence
# if true, it will do the detection everytimme making the model slow
# we keep it false to track the landmarks instead, if above a certain confidence level
# can specify the maximum number of hands and the confidence levels for both detection and tracking
hands = mpHands.Hands()

# this is the function to draw the neccesary coordinates and curves on our images
mpDraw = mp.solutions.drawing_utils

def Coordinates(frame):
    
    # need to convert into rgb as the mediapipe module recognises rgb and not bgr
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # getting the dimensions
    h, w, c = frame.shape
    
    # the if statement is to check whether any hands are captured in the image frame
    if results.multi_hand_landmarks:

        # for looping through all the hands captured in the image frame
        for handLms in results.multi_hand_landmarks:

            x1, x2, y1, y2 = w, 0, h, 0
            # for looping through all the 21 landmark points in a single hand
            # lm gives the coordinates relative to the height and width, 
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                x1 = min(cx, x1)
                x2 = max(cx, x2)
                y1 = min(cy, y1)
                y2 = max(cy, y2)
                # highlighting the index finger tip, mostly going to use this for drawing and etc
                if id == 8:
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

            # drawing the coordinates and curves on the frame
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            cv2.rectangle(frame, (max(0, x1 - 15), min(h, y2 + 15)), (min(w, x2 + 15), max(0, y1 - 15)), (0, 0, 255), 3)

    return frame
