import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
# adjusting values such that only one hand is detected
hands = mpHands.Hands(False, 1, 1, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils

# FindPositions function is used to get the landmark list for the hand 
def FindPositions(frame):
    lm_list = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, c = frame.shape
    
    if results.multi_hand_landmarks:
        # since we restricted it to detections of only one hand
        Hand = results.multi_hand_landmarks[0]

        # looping through the landmarks of the hand and appending the coordinates of the landmarks
        for id, lm in enumerate(Hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])

    return lm_list

# FindGesture function detects which fingers are extended and returns the gesture
def FindGesture(lm_list):
    # id of the tips of the fingers except the thumb
    fingers_id = [8, 12, 16, 20]
    fingers = []

    # checking the relative position of the tips of the fingers with respect a landmarks point below them
    for id in fingers_id:
        if lm_list[id][2] < lm_list[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers
            

def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        lm_list = FindPositions(frame)
        if len(lm_list):
            print(FindGesture(lm_list))
    
        cv2.imshow('Live', frame)

        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()