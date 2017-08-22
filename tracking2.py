# import the necessary packages
from PIL import Image
import numpy as np
import cv2
import pytesseract

# load the games image
image2 = cv2.imread("snapshot8.jpg")
image =cv2.resize(image2, (436, 460))
# find the red color game in the image
image3 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
upper = np.array([130,255,255])
lower = np.array([110,100,100])
mask = cv2.inRange(image, lower, upper)
mask2 = cv2.inRange(image3, lower, upper)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
 
# find contours in the masked image and keep the largest one
(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)

(_, cnts2, _) = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
c2 = max(cnts2, key=cv2.contourArea)
print c
# approximate the contour
peri = cv2.arcLength(c, True)
peri2 = cv2.arcLength(c2, True)

approx = cv2.approxPolyDP(c, 0.05 * peri, True)
approx2 = cv2.approxPolyDP(c2, 0.05 * peri2, True)

x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)

x2,y2,w2,h2 = cv2.boundingRect(c2)
cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),4)


roi=mask[y:y+h,x:x+w]
roiOri = mask[y:y+h,x:x+w]
# print h2
imcrop = image[y:y+h,x:x+w]
imgray = cv2.cvtColor(imcrop,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
# imfinal = imcrop * (roi/255)
# draw a green bounding box surrounding the red game
# cv2.drawContours(image, [approx2], -6, (0, 255, 0), 4)
ret,thresh1 = cv2.threshold(roiOri,127,255,cv2.THRESH_TOZERO_INV)
cv2.imshow("Image", image)
cv2.imshow("Image3", imgray)
# cv2.imshow("Image2", mask)
# cv2.imshow("Image4", mask2)
# cv2.imwrite("aa.jpg", thresh)
# text = pytesseract.image_to_string(Image.open('aa.jpg'))
# print text
# print 'a'
# cv2.imshow("Image3", gray)
# cv2.imshow("Image4", edged)
cv2.waitKey(0)
