import cv2

class Display(object):
    def __init__(self):
        self.W = 1920//2
        self.H = 1080//2

    def show(self, img):
        img = cv2.resize(img, (self.W, self.H))
        cv2.imshow('frame', img)