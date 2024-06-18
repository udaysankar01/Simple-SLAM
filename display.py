import cv2

class Display(object):
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def show(self, img):
        img = cv2.resize(img, (self.W, self.H))
        cv2.imshow('frame', img)