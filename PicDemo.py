import cv2 as cv
import numpy as np
import matplotlib as plt

IMG=cv.imread('Pic/oRG.png',0)

# IMG_GRAY=cv.cvtColor(IMG,cv.COLOR_BGR2GRAY)
# cv.imshow('IMGGRAY',IMG_GRAY)
#
MI=cv.imread('Pic/MI.jpg',0)
orb=cv.ORB_create()
kp1,des1=orb.detectAndCompute(IMG,None)
kp2,des2=orb.detectAndCompute(MI,None)
bf=cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(IMG,kp1,MI,kp2,matches[:10],None, flags=2)
cv.imshow('Result',img3)

# cv.imshow('mi',MI)
# w,h=MI.shape[::-1]
#
#
# res=cv.matchTemplate(IMG_GRAY,MI,cv.TM_CCOEFF_NORMED)
# threshold=0.2
# loc=np.where(res >=threshold)
#
# for pt in zip(*loc[::-1]):
#         cv.rectangle(IMG,pt,(pt[0]+w,pt[1]+h),(0,0,0),5)
# cv.imshow('detected',IMG)

if cv.waitKey(0):
        cv.destroyAllWindows()
