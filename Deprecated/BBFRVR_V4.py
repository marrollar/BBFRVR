from bbProcessing import FindBallAndHoop
import pyautogui
import cv2
import numpy as np
import time
import threading
import matplotlib

matplotlib.use('TkAgg')
import multiprocessing
from tkinter import *
import matplotlib.pyplot as plt

printLock = threading.Lock()
findBH = FindBallAndHoop()


# class Plotter(threading.Thread):
#
#     def __init__(self, queue, *args):
#         threading.Thread.__init__(self)
#         self.queue = queue
#         self.daemon = True
#         self.receiveParams = args[0]
#         self.setupPlot()
#
#     def run(self):
#         print(threading.current_thread().getName(), self.receiveParams)
#         val = self.queue.get()
#         self.updatePlot(val)
#
#     def setupPlot(self):
#         plt.ion()
#         self.fig, self.ax = plt.subplots()
#         self.x, self.y = [], []
#         self.sc = self.ax.scatter(self.x, self.y)
#         plt.draw()
#
#     def updatePlot(self, point):
#         if self.receiveParams:
#             with printLock:
#                 self.x.append(point[0])
#                 self.y.append(point[1])
#                 self.sc.set_offsets(np.c_[self.x, self.y])
#                 self.fig.canvas.draw_idle()
#                 plt.pause(0.1)
#
# class Shooter(threading.Thread):
#
#     def __init__(self):


def calculateVelocity(p1, p2, dt):
    velX = (p2[0] - p1[0]) / dt
    velY = (p2[1] - p2[1]) / dt
    return velX, velY


def setupPlot():
    # global line, ax, canvas
    # fig = matplotlib.figure.Figure()
    # ax = fig.add_subplot(1, 1, 1)
    # canvas = FigureCanvasTkAgg(fig, master=window)
    # canvas.show()
    # canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    # canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    # line, = ax.plot([1, 2, 3], [1, 2, 10])
    global x, y, sc, fig
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    plt.draw()


def updatePlot(q):
    try:
        result = q.get_nowait()

        print("Queue data: {}".format(result))

        x.append(result[0])
        y.append(result[1])

        print("X: {}".format(x))
        print ("Y: {}".format(y))
        sc.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.1)

        # if result != 'Q':
        #     print(result)
        #     line.set_ydata([1, result, 10])
        #     ax.draw_artist(line)
        #     canvas.draw()
        #     window.after(500, updatePlot, q)
        # else:
        #     print("Done")
    except:
        print("Empty")
        t = threading.Timer(0.5, updatePlot)
        t.start()


def shoot(q, startTime):
    source = pyautogui.screenshot()
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    source = source[180:1028, 294:770]

    findBH.process(source)

    ballMoment, hoopMoment = findBH.getMoments()

    centerB = (int((ballMoment["m10"] / ballMoment['m00'])),
               int((ballMoment["m01"] / ballMoment["m00"])))

    centerH = (int((hoopMoment["m10"] / hoopMoment['m00'])),
               int((hoopMoment["m01"] / hoopMoment["m00"])))
    centerH = (centerH[0], centerH[1] - 75)

    q.put(centerB)

    lastHoopPos = []
    while time.time() - startTime < 2:
        # print("Last Hoop Pos: %d" % (len(lastHoopPos)))
        lastHoopPos.append([centerH, time.time()])

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
    pyautogui.dragTo(halfway[0] + 294, halfway[1] + 180, 0.1, button="left")
    time.sleep(3)


if __name__ == "__main__":
    q = multiprocessing.Queue()
    shooter = multiprocessing.Process(None, shoot, args=(q, time.time(),))
    setupPlot()

    cv2.namedWindow("Get Keyboard  Input")

    run = False
    while True:
        keyPressed = cv2.waitKey(33)
        if keyPressed == ord("q"):
            cv2.destroyAllWindows()
            plt.close("all")
            shooter.terminate()
            shooter.join()
            sys.exit()

        elif keyPressed == ord("g"):
            if run == False:
                print("RUNNING")
                run = True
                shooter.start()
                # updatePlot(q)

            else:
                print("STOPPING")
                run = False
                shooter.terminate()
                shooter.join()
