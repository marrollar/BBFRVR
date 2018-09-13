import cv2
import numpy as np

lower = np.array([10, 130, 157])
upper = np.array([21, 255, 255])


class FindBallAndHoop:

    def process(self, source):
        self.originalImg = source

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

        if len(sortedAreas) >= 2:
            processedImg = ProcessedImage(self.originalImg, sortedAreas[0][0], cv2.moments(sortedAreas[0][1]), sortedAreas[1][0], cv2.moments(sortedAreas[1][1]))
            return processedImg


class ProcessedImage:

    def __init__(self, oImg, ballC, ballM, hoopC, hoopM):
        self.img = oImg
        self.ballContour = ballC
        self.ballMoment = ballM
        self.hoopContour = hoopC
        self.hoopMoment = hoopM

    def getContours(self):
        return self.ballContour, self.hoopContour

    def getMoments(self):
        return self.ballMoment, self.hoopMoment

    def getImg(self):
        return self.img
