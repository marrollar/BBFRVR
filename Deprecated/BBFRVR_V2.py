import cv2
import pyautogui
import numpy as np
import sys
import time

lower = np.array([10, 130, 157])
upper = np.array([21, 255, 255])
# cv2.imwrite("bbfrvr.jpg", image)
# image = cv2.imread("bbfrvr.jpg")

lastHoopPos = None
cv2.namedWindow("Click Recorder")


def calculateVelocity(p1, p2, dt):
    velX = (p2[0] - p1[0]) / dt
    velY = (p2[1] - p2[1]) / dt
    return velX, velY


startTime = time.time()
run = False

while True:

    source = pyautogui.screenshot()
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    # source = source[:, 0:(int(source.shape[1] / 2))]
    source = source[180:1028, 294:770]

    image = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image, lower, upper)

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    kernel3 = np.ones((15, 15), np.uint8)

    erode = cv2.erode(mask, kernel=kernel1)
    dilate = cv2.dilate(erode, kernel=kernel1)
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel3)
    erode2 = cv2.dilate(close, kernel=kernel2)

    fStream, validContours, hierarchy = cv2.findContours(erode2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    largestCs = []
    cAreas = []

    for c in validContours:
        c = c.astype('int')
        cArea = cv2.contourArea(c)

        if cArea > 1000:
            largestCs.append(c)
            cAreas.append(cArea)

    sortedAreas = sorted(zip(cAreas, largestCs), key=lambda l: l[0], reverse=True)

    for i in range(len(sortedAreas)):
        try:
            cMoments = cv2.moments(sortedAreas[i][1])
            centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                           int((cMoments["m01"] / cMoments["m00"])))

            cv2.circle(image, (centerPoint[0], centerPoint[1]), 4, (0, 255, 0), -1)
        except ZeroDivisionError:
            pass

    largestCM = cv2.moments(sortedAreas[0][1])
    smallestCM = cv2.moments(sortedAreas[1][1])

    centerB = (int((largestCM["m10"] / largestCM['m00'])),
               int((largestCM["m01"] / largestCM["m00"])))

    centerH = (int((smallestCM["m10"] / smallestCM['m00'])),
               int((smallestCM["m01"] / smallestCM["m00"])))
    centerH = (centerH[0], centerH[1] - 75)


    keyPressed = cv2.waitKey(33)

    if keyPressed == ord("q"):
        cv2.destroyAllWindows()
        sys.exit()
    elif keyPressed == ord("g"):
        if run:
            print("STOPPING")
            run = False
        else:
            print("RUNNING")
            run = True

    if run:
        if lastHoopPos is not None:

            print("\n----------------------------")

            print (lastHoopPos)

            dt = lastHoopPos[1] - time.time()
            velX, velY = calculateVelocity(lastHoopPos[0], centerH, dt)
            print(velX, velY)

            halfway = [((centerB[0] + centerH[0]) / 2) + (velX * dt), ((centerB[1] + centerH[1]) / 2) + (velY * dt)]
            halfImageX = image.shape[1] / 2
            halfImageY = image.shape[0] / 2

            cv2.circle(source, centerB, 5, (0, 255, 0), -1)
            cv2.circle(source, centerH, 5, (0, 0, 255), -1)
            cv2.line(source, (int(centerB[0]), 0), (int(centerB[0]), image.shape[0]), (255, 0, 0))
            cv2.line(source, (0, int(centerH[1])), (image.shape[1], int(centerH[1])), (0, 255, 0))
            cv2.line(source, (0, int(halfImageY)), (image.shape[1], int(halfImageY)), (0, 0, 255))

            print("Hoop: {}".format(centerH))
            print("Ball: {}".format(centerB))
            print("Halfway X - Original: {}".format(halfway))

            cv2.circle(source, (int(halfway[0]), int(halfway[1])), 5, (255, 0, 0), -1)

            if (centerH[0] < halfImageX):
                percentFromCenterX = (centerB[0] - centerH[0]) / centerB[0]
                print("Percent from Center X: {}".format(percentFromCenterX))
                originalX = halfway[0]
                halfway[0] += (centerB[0] - halfway[0]) * percentFromCenterX
                halfway[0] -= (centerB[0] - halfway[0]) * percentFromCenterX
                # halfway[0] += (originalX - halfway[0]) * percentFromSide


            elif (centerH[0] > halfImageX):
                percentFromCenterX = (centerH[0] - centerB[0]) / centerB[0]
                print("Percent from Center X: {}".format(percentFromCenterX))
                originalX = halfway[0]
                halfway[0] -= (halfway[0] - centerB[0]) * percentFromCenterX
                halfway[0] += (halfway[0] - centerB[0]) * percentFromCenterX
                # halfway[0] -= (halfway[0] - originalX) * percentFromSide

            # if (centerH[1] < halfImageY):
            #     percentFromCenterY = (halfImageY - centerH[1]) / halfImageY
            #     halfway[1] += (halfway[1] - halfImageY) * percentFromCenterY
            #     halfway[1] -= (halfway[1] - halfImageY) * percentFromCenterY
            #
            #
            # elif (centerH[1] > halfImageY):
            #     percentFromCenterY = (centerH[1] - halfImageY) / halfImageY
            #     halfway[1] -= (halfImageY - halfway[1]) * percentFromCenterY
            #     halfway[1] += (halfImageY - halfway[1]) * percentFromCenterY


            print("Halfway - Shifted: {}".format(halfway))
            cv2.circle(source, (int(halfway[0]), int(halfway[1])), 5, (0, 255, 255), -1)
            cv2.imshow("image", source)
            cv2.moveWindow("image", 1000, 0)

            pyautogui.moveTo(centerB[0] + 294, centerB[1] + 180)
            pyautogui.dragTo(halfway[0] + 294, halfway[1] + 180, 0.1, button="left")
            time.sleep(3)
            lastHoopPos = [centerH, time.time()]
            startTime = time.time()
        else:
            lastHoopPos = [centerH, time.time()]

