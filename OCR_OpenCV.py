
import cv2 as cv
import numpy as np

image = cv.imread("Pic/112.jpg")
cv.imshow("input", image)
result = image.copy()
detector = cv.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
boxes, scores = detector.detect(image);
threshold = 0.5
for r in range(np.shape(boxes)[0]):
    if scores[r] > threshold:
        rect = boxes[r]
        cv.rectangle(result, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

cv.imshow("Text detection result", result)
cv.waitKey()

cv.waitKey(0)
cv.destroyAllWindows()
