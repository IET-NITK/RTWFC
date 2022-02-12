import cv2
import mediapipe as mp

class HandDetector:

    def __init__(self, mode = False, maxHands = 1, modcomplex = 1, DetectCon = 0.5, TrackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modcomplex = modcomplex
        self.DetectCon = DetectCon
        self.TrackCon = TrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modcomplex, self.DetectCon, self.TrackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, frame, draw = True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def FindPositions(self, frame, HandNo = 0, draw = True):
        lm_list = []
        h, w, c = frame.shape

        if self.results.multi_hand_landmarks:

            Hand = self.results.multi_hand_landmarks[HandNo]
            for id, lm in enumerate(Hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

        return lm_list

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.FindHands(frame, True)
        lm_list = detector.FindPositions(frame, 0, True)

        if len(lm_list):
            print(lm_list[8])
    
        cv2.imshow('Live', frame)

        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()