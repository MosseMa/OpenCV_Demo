import subprocess

import pytesseract as ocr
import cv2 as cv
import numpy as np


img= cv.imread('Pic/1212.jpg',0)
# alpha and beta convert to adjust contrast
new_image = cv.convertScaleAbs(img, alpha=0.7, beta=80)
cv.imshow('contrast',new_image)
# gamma convert to adjust brightness
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 5) * 255.0, 0, 255)
cvt = cv.LUT(img, lookUpTable)
res=ocr.image_to_string(cvt)
print(res)
cv.imshow('img',cvt)

if cv.waitKey(0):
        cv.destroyAllWindows()
