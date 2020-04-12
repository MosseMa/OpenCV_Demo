import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

cam = cv.VideoCapture(0)
while(True):
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()

