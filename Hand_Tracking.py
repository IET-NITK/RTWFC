import cv2
import mediapipe as mp

# compulsory to implement mediapipe functions
mpHands = mp.solutions.hands

# has five parameters, static_image_mode = false, max_hands, model_complexity, min_detection_confidence, min_tracking_confidence
# if true, it will do the detection everytime making the model slow
# we keep it false to track the landmarks instead, if above a certain confidence level
# can specify the maximum number of hands and the confidence levels for both detection and tracking
hands = mpHands.Hands()

# this is the function to draw the neccesary coordinates and curves on our images
mpDraw = mp.solutions.drawing_utils

def FindHands(frame):
    # need to convert into rgb as the mediapipe module recognises rgb and not bgr
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # getting the dimensions
    h, w, c = frame.shape
    
    # the if statement is to check whether any hands are captured in the image frame
    if results.multi_hand_landmarks:

        # for looping through all the hands captured in the image frame
        for handLms in results.multi_hand_landmarks:

            # for looping through all the 21 landmark points in a single hand
            # lm gives the coordinates relative to the height and width, 
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

            # drawing the coordinates and curves on the frame
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = FindHands(frame)
    
        cv2.imshow('Live', frame)

        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()