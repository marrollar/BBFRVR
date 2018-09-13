import pyautogui
import cv2
import numpy as np
import multiprocessing


class ImageGrabber(multiprocessing.Process):

    def __init__(self, q, imgQ):
        multiprocessing.Process.__init__(self)
        self.queue = q
        self.imgQueue = imgQ

    def run(self):
        while True:
            task = self.queue.get()
            if task is None:
                print("Image Grabber SHUTTING DOWN")
                self.queue.task_done()
                break
            elif task:
                source = pyautogui.screenshot()
                source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
                source = source[180:1028, 294:770]

                self.queue.task_done()
                self.imgQueue.put(source)
        return
