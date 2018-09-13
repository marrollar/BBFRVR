from bbProcessing import FindBallAndHoop
import pyautogui
import cv2
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


def calculateVelocity(p1, p2, dt):
    velX = (p2[0] - p1[0]) / dt
    velY = (p2[1] - p2[1]) / dt
    return velX, velY


findBH = FindBallAndHoop()
run = False
plotBall = False
plt.ion()
plt.draw()

cv2.namedWindow("Get Keyboard Input")

lastHoopPos = []
startTime = 0
while True:
    source = pyautogui.screenshot()
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    # source = source[:, 0:(int(source.shape[1] / 2))]
    source = source[180:1028, 294:770]

    processed = findBH.process(source)

    keyPressed = cv2.waitKey(33)
    #
    if keyPressed == ord("q"):
        cv2.destroyAllWindows()
        sys.exit()
    elif keyPressed == ord("g"):
        if run:
            print("STOPPING")
            run = False
        else:
            print("RUNNING")
            startTime = time.time()
            run = True
    elif keyPressed == ord("p"):
        if plotBall == False:
            print("PLOTTING")
            plotBall = True
        else:
            print("NO PLOTTING")
            plotBall = False

    if run:

        ballMoment, hoopMoment = processed.getMoments()

        centerB = (int((ballMoment["m10"] / ballMoment['m00'])),
                   int((ballMoment["m01"] / ballMoment["m00"])))

        centerH = (int((hoopMoment["m10"] / hoopMoment['m00'])),
                   int((hoopMoment["m01"] / hoopMoment["m00"])))
        centerH = (centerH[0], centerH[1] - 75)

        if time.time() - startTime < 2:
            print(len(lastHoopPos))
            lastHoopPos.append([centerH, time.time()])

        else:
            print("\n----------------------------")
            print(lastHoopPos)

            sum_dx = 0
            sum_dy = 0
            for i in range(len(lastHoopPos) - 1):
                dx, dy = calculateVelocity(lastHoopPos[i][0], lastHoopPos[i + 1][0], lastHoopPos[i + 1][1] - lastHoopPos[i][1])

                sum_dx += dx
                sum_dy += dy

            average_dx = sum_dx / len(lastHoopPos)
            average_dy = sum_dy / len(lastHoopPos)

            print(average_dx, average_dy)

            halfway = [((centerB[0] + centerH[0]) / 2) + (average_dx), ((centerB[1] + centerH[1]) / 2) + (average_dy)]
            originalHW = halfway
            halfImageX = source.shape[1] / 2
            halfImageY = source.shape[0] / 2

            if (centerH[0] < halfImageX):
                percentFromCenterX = (centerB[0] - centerH[0]) / centerB[0]
                print("Percent from Center X: {}".format(percentFromCenterX))
                originalX = halfway[0]
                halfway[0] += (centerB[0] - halfway[0]) * percentFromCenterX
                halfway[0] -= (centerB[0] - halfway[0]) * percentFromCenterX


            elif (centerH[0] > halfImageX):
                percentFromCenterX = (centerH[0] - centerB[0]) / centerB[0]
                print("Percent from Center X: {}".format(percentFromCenterX))
                originalX = halfway[0]
                halfway[0] -= (halfway[0] - centerB[0]) * percentFromCenterX
                halfway[0] += (halfway[0] - centerB[0]) * percentFromCenterX

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

            # -------------------------------------------------------------------------#
            # -------------------------------------------------------------------------#
            # -------------------------------------------------------------------------#

            print("Hoop: {}".format(centerH))
            print("Ball: {}".format(centerB))
            print("Halfway X - Original: {}".format(originalHW))
            print("Halfway - Shifted: {}".format(halfway))

            cv2.circle(source, centerB, 5, (0, 255, 0), -1)  # Center of Ball
            cv2.circle(source, centerH, 5, (0, 0, 255), -1)  # Center of Hoop shifted up to the top backboard line
            cv2.line(source, (int(centerB[0]), 0), (int(centerB[0]), source.shape[0]), (255, 0, 0))  # Vertical line on center of ball
            cv2.line(source, (0, int(centerH[1])), (source.shape[1], int(centerH[1])), (0, 255, 0))  # Horizontal line on center of hoop shifted up to top backboard line
            cv2.line(source, (0, int(halfImageY)), (source.shape[1], int(halfImageY)), (0, 0, 255))  # Horizontal line on actual center of image

            cv2.circle(source, (int(originalHW[0]), int(originalHW[1])), 5, (255, 0, 0), -1)  # Halfway before processing
            cv2.circle(source, (int(halfway[0]), int(halfway[1])), 5, (0, 255, 255), -1)  # Halfway after processing

            magnitude = np.sqrt((halfway[1] - centerB[1]) ** 2 + (halfway[0] - centerB[0]) ** 2)
            endX = halfway[0] + (halfway[0] - centerB[0]) / magnitude * 1000
            endY = halfway[1] + (halfway[1] - centerB[1]) / magnitude * 1000

            # C.x = B.x + (B.x - A.x) / lenAB * length;
            # C.y = B.y + (B.y - A.y) / lenAB * length;

            cv2.line(source, (int(centerB[0]), int(centerB[1])), (int(endX), int(endY)), (255, 255, 255))

            cv2.imshow("image", source)
            cv2.moveWindow("image", 1000, 0)

            # -------------------------------------------------------------------------#
            # -------------------------------------------------------------------------#
            # -------------------------------------------------------------------------#

            pyautogui.moveTo(centerB[0] + 294, centerB[1] + 180)
            pyautogui.dragTo(halfway[0] + 294, halfway[1] + 180, 0.2, button="left")

            ballPos = []

            if plotBall:
                startPlotTime = time.time()
                timeAxis = []
                plt.cla()

                while time.time() - startPlotTime < 3:
                    source = pyautogui.screenshot()
                    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
                    # source = source[:, 0:(int(source.shape[1] / 2))]
                    source = source[180:1028, 294:770]

                    findBH.process(source)
                    ballMoment, hoopMoment = findBH.getMoments()

                    centerB = (int((ballMoment["m10"] / ballMoment['m00'])),
                               int((ballMoment["m01"] / ballMoment["m00"])))
                    ballPos.append(centerB[1])
                    timeAxis.append(time.time() - startPlotTime)

                xp = np.linspace(0, 3, 100)
                poly = np.polyfit(timeAxis[:4], ballPos[:4], 2)
                poly1D = np.poly1d(poly)
                print("Polynomial: {}".format(poly1D))
                print(poly[0])
                with open("polynomialFits.txt", 'a') as outFile:
                    strList = "".join((str(int(i)) + "," for i in poly))
                    outFile.write(strList + "\n")
                    outFile.close()

                plt.plot(timeAxis[:5], ballPos[:5], '.', xp, poly1D(xp), '-')
                plt.pause(0.1)
                plt.ylim(100, 800)
            else:
                time.sleep(3)

            startTime = time.time()
            lastHoopPos.clear()
            ballPos.clear()
