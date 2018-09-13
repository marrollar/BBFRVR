import multiprocessing

import cv2
import sys

from bbProcessing import FindBallAndHoop
from bbProcessing import ProcessedImage
from bbScreenshot import ImageGrabber
import pyautogui
import time
import numpy as np


# class ImageAnalyzer(multiprocessing.Process):
#
#     def __init__(self, imgQ):
#         multiprocessing.Process.__init__(self)
#         self.imgQueue = imgQ
#         self.bhFinder = FindBallAndHoop()
#
#     def run(self):
#         while True
#             img = self.imgQueue.get()
#
#             if img is None:
#                 print("Image Analyzer SHUTTING DOWN")
#
#
#

def calcVel(p1, p2, dt):
    velX = (p2[0] - p1[0]) / dt
    velY = (p2[1] - p2[1]) / dt
    return velX, velY


def calcDist(p1, p2):
    return np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)


def analyzeImage(imgQueue, processedQueue):
    while True:
        img = imgQueue.get()

        if img is None:
            print("Image analyzer STOPPING")
            imgQueue.task_done()

        processedImg = bhFinder.process(img)

        imgQueue.task_done()
        processedQueue.put(processedImg)


if __name__ == '__main__':

    global bhFinder
    bhFinder = FindBallAndHoop()

    commandsQ = multiprocessing.JoinableQueue()
    imgQueue = multiprocessing.JoinableQueue()
    processedQueue = multiprocessing.Queue()
    allQueues = [commandsQ, imgQueue, processedQueue]

    imgGrabber = ImageGrabber(commandsQ, imgQueue)
    analyzer = multiprocessing.Process(target=analyzeImage, args=(imgQueue, processedQueue))
    imgGrabber.start()
    analyzer.start()

    cv2.namedWindow("Key Cap Window")

    running = False
    getCap = True
    prevHoop = []

    gravity = 3066  # pixels/s^2
    timeToPeak = 0.6994357031207934 # s
    peakHeight = 174.5992187421771 # px
    ballTraj = np.poly1d([1533, -2164, 945])

    xBound = [70, 390]

    prevVX = 0

    while True:
        keyPressed = cv2.waitKey(33)

        if keyPressed == ord("r"):
            if not running:
                print("Shooting GO")
                running = True
            elif running:
                print("Shooting STOPPED")
                running = False

        # elif keyPressed == ord("g"):
        #     print("Shooting ONCE")
        #     running = False
        #     commandsQ.put("GetCap")
        #
        #     processed = processedQueue.get()
        #     if processed is not None:
        #         print(processed)
        #

        elif keyPressed == ord("q"):
            for q in allQueues:
                q.put(None)
                q.join()

            imgGrabber.terminate()
            imgGrabber.join()

            cv2.destroyAllWindows()
            sys.exit()

        if running:
            commandsQ.put(getCap)

            processed = processedQueue.get()
            if processed is not None:
                getCap = False
                source = processed.getImg()

                ballMoment, hoopMoment = processed.getMoments()

                centerB = (int((ballMoment["m10"] / ballMoment['m00'])),
                           int((ballMoment["m01"] / ballMoment["m00"])))

                centerH = (int((hoopMoment["m10"] / hoopMoment['m00'])),
                           int((hoopMoment["m01"] / hoopMoment["m00"])))
                centerH = (centerH[0], centerH[1] - 75)

                if len(prevHoop) == 0:
                    prevHoop = ([centerH, time.time()])
                    getCap = True

                else:
                    dyFromPeak = centerH[1] - peakHeight
                    tToHoop = np.sqrt(2 * dyFromPeak / gravity) + timeToPeak

                    velX, velY = calcVel(centerH, prevHoop[0], time.time() - prevHoop[1])
                    velX = -velX

                    if ((velX < 0 and velX + prevVX > 0) or (velX > 0 and velX + prevVX < 0)):
                        avgX = (centerH[0] + prevHoop[0][0]) / 2
                        avgY = (centerH[1] + prevHoop[0][1]) / 2

                    xMagnitude = velX * tToHoop
                    dxCoord = [centerH[0] + xMagnitude, centerH[1]]

                    if dxCoord[0] < xBound[0]:
                        dxCoord[0] = xBound[0]
                        distToBound = calcDist(centerH, (xBound[0], dxCoord[1]))
                        reboundMag = xMagnitude + distToBound  # xMag is negative due to velX being negative

                        dxCoord = [dxCoord[0] - reboundMag, dxCoord[1]]
                        cv2.line(source, (xBound[0], dxCoord[1]), tuple((int(c) for c in dxCoord)), (255, 0, 0), 4)


                    elif dxCoord[0] > xBound[1]:
                        dxCoord[0] = xBound[1]
                        distToBound = np.abs(xBound[1] - centerH[0])
                        reboundMag = xMagnitude - distToBound # xMag is pos due to velX being pos

                        dxCoord = [dxCoord[0] - reboundMag, dxCoord[1]]
                        cv2.line(source, (xBound[1], dxCoord[1]), tuple((int(c) for c in dxCoord)), (255, 0, 0), 4)


                    cv2.circle(source, tuple(int(c) for c in dxCoord), 7, (0, 69, 255), -1) # Draw initial vector end point
                    cv2.line(source, tuple(int(c) for c in centerH), tuple(int(c) for c in dxCoord), (0, 69, 255), 2) # Draw inital vector line

                    cv2.circle(source, centerH, 5, (0, 255, 0), -1) # Draw circle at center of hoop
                    cv2.circle(source, prevHoop[0], 5, (0, 0, 255), -1) # Draw circle at previous hoop position
                    cv2.putText(source, "({}, {})".format(int(velX), int(velY)), (centerH[0], centerH[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Display x and y velocity

                    cv2.line(source, (xBound[0], 0), (xBound[0], 848), (0, 0, 255), 2, cv2.LINE_AA) # Draw x boundaries
                    cv2.line(source, (xBound[1], 0), (xBound[1], 848), (0, 0, 255), 2, cv2.LINE_AA)

                    prevVX = velX
                    cv2.imshow("image", source)
                    cv2.moveWindow("image", 1000, 0)

                    pyautogui.moveTo(centerB[0] + 294, centerB[1] + 180)
                    pyautogui.dragTo(dxCoord[0] + 294, dxCoord[1] + 180, 0.2, button="left")
                    time.sleep(3)
                    getCap = True
                    prevHoop = []

                # if time.time() - startTime < 2:
                #     print(len(lastHoopPos))
                #     lastHoopPos.append([centerH, time.time()])
                #
                # else:
                #     print("\n----------------------------")
                #     print(lastHoopPos)
                #
                #     sum_dx = 0
                #     sum_dy = 0
                #     for i in range(len(lastHoopPos) - 1):
                #         dx, dy = calcVel(lastHoopPos[i][0], lastHoopPos[i + 1][0], lastHoopPos[i + 1][1] - lastHoopPos[i][1])
                #
                #         sum_dx += dx
                #         sum_dy += dy
                #
                #     average_dx = sum_dx / len(lastHoopPos)
                #     average_dy = sum_dy / len(lastHoopPos)
                #
                #     print(average_dx, average_dy)
                #
                #     halfway = [((centerB[0] + centerH[0]) / 2) + (average_dx), ((centerB[1] + centerH[1]) / 2) + (average_dy)]
                #     originalHW = halfway
                #     halfImageX = source.shape[1] / 2
                #     halfImageY = source.shape[0] / 2
                #
                #     if (centerH[0] < halfImageX):
                #         percentFromCenterX = (centerB[0] - centerH[0]) / centerB[0]
                #         print("Percent from Center X: {}".format(percentFromCenterX))
                #         originalX = halfway[0]
                #         halfway[0] += (centerB[0] - halfway[0]) * percentFromCenterX
                #         halfway[0] -= (centerB[0] - halfway[0]) * percentFromCenterX
                #
                #
                #     elif (centerH[0] > halfImageX):
                #         percentFromCenterX = (centerH[0] - centerB[0]) / centerB[0]
                #         print("Percent from Center X: {}".format(percentFromCenterX))
                #         originalX = halfway[0]
                #         halfway[0] -= (halfway[0] - centerB[0]) * percentFromCenterX
                #         halfway[0] += (halfway[0] - centerB[0]) * percentFromCenterX

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

                # print("Hoop: {}".format(centerH))
                # print("Ball: {}".format(centerB))
                # print("Halfway X - Original: {}".format(originalHW))
                # print("Halfway - Shifted: {}".format(halfway))
                #
                # cv2.circle(source, centerB, 5, (0, 255, 0), -1)  # Center of Ball
                # cv2.circle(source, centerH, 5, (0, 0, 255), -1)  # Center of Hoop shifted up to the top backboard line
                # cv2.line(source, (int(centerB[0]), 0), (int(centerB[0]), source.shape[0]), (255, 0, 0))  # Vertical line on center of ball
                # cv2.line(source, (0, int(centerH[1])), (source.shape[1], int(centerH[1])), (0, 255, 0))  # Horizontal line on center of hoop shifted up to top backboard line
                # cv2.line(source, (0, int(halfImageY)), (source.shape[1], int(halfImageY)), (0, 0, 255))  # Horizontal line on actual center of image
                #
                # cv2.circle(source, (int(originalHW[0]), int(originalHW[1])), 5, (255, 0, 0), -1)  # Halfway before processing
                # cv2.circle(source, (int(halfway[0]), int(halfway[1])), 5, (0, 255, 255), -1)  # Halfway after processing
                #
                # magnitude = np.sqrt((halfway[1] - centerB[1]) ** 2 + (halfway[0] - centerB[0]) ** 2)
                # endX = halfway[0] + (halfway[0] - centerB[0]) / magnitude * 1000
                # endY = halfway[1] + (halfway[1] - centerB[1]) / magnitude * 1000
                #
                # # C.x = B.x + (B.x - A.x) / lenAB * length;
                # # C.y = B.y + (B.y - A.y) / lenAB * length;
                #
                # cv2.line(source, (int(centerB[0]), int(centerB[1])), (int(endX), int(endY)), (255, 255, 255))
                #

                # -------------------------------------------------------------------------#
                # -------------------------------------------------------------------------#
                # -------------------------------------------------------------------------#

                # pyautogui.moveTo(centerB[0] + 294, centerB[1] + 180)
                # pyautogui.dragTo(halfway[0] + 294, halfway[1] + 180, 0.2, button="left")
                # startTime = time.time()
                # lastHoopPos.clear()
                # ballPos.clear()
