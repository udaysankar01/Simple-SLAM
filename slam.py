#!/usr/bin/env python3

import cv2 
import numpy as np

W = 1920//2
H = 1080//2

def process_image(img):
    img = cv2.resize(img, (W, H))
    cv2.imshow('frame', img)
    return img

def main():
    cap = cv2.VideoCapture('videos/test.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = process_image(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()