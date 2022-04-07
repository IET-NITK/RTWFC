from turtle import home
import cv2
import os
import numpy as np
from PIL import Image
import HandTracking_GestureRecognition_Module as hgm

colorsPath = "NavBar/Colors"
homepagePath = "NavBar/Homepage"
sizesPath = "NavBar/Sizes"
imListColors = os.listdir(colorsPath)
imListHomepage = os.listdir(homepagePath)
imListSizes = os.listdir(sizesPath)
colors = []
homepage = []
sizes = []

for imPath in imListColors:
    image = cv2.imread(f'{colorsPath}/{imPath}')
    colors.append(image)

for imPath in imListHomepage:
    image = cv2.imread(f'{homepagePath}/{imPath}')
    homepage.append(image)

for imPath in imListSizes:
    image = cv2.imread(f'{sizesPath}/{imPath}')
    sizes.append(image)

def drawOnFeed(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, ImgInv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    ImgInv = cv2.cvtColor(ImgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, ImgInv)
    frame = cv2.bitwise_or(frame, canvas)

    return frame

def main():
    width, height = 1280, 720
    brushColor = [(0, 0, 255), (0, 255, 0), (255, 0, 80)]
    brushSize = [10, 20, 30]
    eraserSize = [25, 45, 60]
    currNavBar, currNavBarid, currColor, currBrushsize, currEraserSize = homepage[0], 0, brushColor[2], brushSize[1], eraserSize[1]
    canvas = np.zeros((height, width, 3), dtype = 'uint8')

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

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
            xm, ym = lm_list[12][1:]

            # index finger
            if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                if xp == 0 and yp == 0:
                    xp, yp = xi, yi
                
                cv2.line(canvas, (xp, yp), (xi, yi), currColor, currBrushsize)
                xp, yp = xi, yi

            # index + middle fingers
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                xp, yp = 0, 0

                if currNavBarid == 0:
                    if ym < 100:
                        if xm > 100 and xm < 280:
                            currNavBar, currNavBarid = colors[0], 1

                        elif xm > 400 and xm < 620:
                            currNavBar, currNavBarid = sizes[1], 2

                        elif xm > 780 and xm < 940:
                            currNavBar, currNavBarid = sizes[0], 3

                elif currNavBarid == 1:
                    if ym < 100:
                        if xm > 100 and xm < 280:
                            currNavBar, currColor = colors[0], brushColor[2]

                        elif xm > 400 and xm < 620:
                            currNavBar, currColor = colors[2], brushColor[0]

                        elif xm > 780 and xm < 940:
                            currNavBar, currColor = colors[1], brushColor[1]

                        elif xm > 1080 and xm < 1200:
                            currNavBar, currNavBarid = homepage[0], 0

                elif currNavBarid == 2:
                    if ym < 100:
                        if xm > 100 and xm < 280:
                            currNavBar, currBrushsize = sizes[2], brushSize[0]

                        elif xm > 400 and xm < 620:
                            currNavBar, currBrushsize = sizes[1], brushSize[1]

                        elif xm > 780 and xm < 940:
                            currNavBar, currBrushsize = sizes[0], brushSize[2]

                        elif xm > 1080 and xm < 1200:
                            currNavBar, currNavBarid = homepage[0], 0
                
                elif currNavBarid == 3:
                    if ym < 100:
                        if xm > 100 and xm < 280:
                            currNavBar, currEraserSize = sizes[2], eraserSize[0]

                        elif xm > 400 and xm < 620:
                            currNavBar, currEraserSize = sizes[1], eraserSize[1]

                        elif xm > 780 and xm < 940:
                            currNavBar, currEraserSize = sizes[0], eraserSize[2]

                        elif xm > 1080 and xm < 1200:
                            currNavBar, currNavBarid = homepage[0], 0

            
            # index + middle + ring fingers
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                xp, yp = 0, 0

                # img = Image.fromarray(canvas)
                # img.save("drawing.png")

            # index + middle + ring + pbrushColory fingers
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                cv2.circle(frame, (xm, ym), currEraserSize, (0, 0, 0), -1)
                cv2.circle(canvas, (xm, ym), currEraserSize, (0, 0, 0), -1)
                xp, yp = 0, 0
            
            else:
                xp, yp = 0, 0

        frame = drawOnFeed(frame, canvas)
        frame[0:100, 0:1280] = currNavBar
        cv2.imshow('Live', frame)

        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
