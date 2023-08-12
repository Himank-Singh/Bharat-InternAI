import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# read and resize image to the required size
img = cv2.imread('image.jpg')
img = imutils.resize(img, width=500)
cv2.imshow("Original Image", img)

# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale", gray)

# blur to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("Noise reduced", gray)

# perform edge detection
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("Edged", edged)



# find contours in the edged image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]

NumberPlateCnt = None 
count = 0
# loop over contours
for c in cnts:
	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the approximated contour has four points, then assume that screen is found
    if len(approx) == 4:  
        NumberPlateCnt = approx 
        break

# mask the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
final_img = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
final_img = cv2.bitwise_and(img,img,mask=mask)
cv2.namedWindow("Number Plate",cv2.WINDOW_NORMAL)
cv2.imshow("Number Plate",final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


 
# configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# run tesseract OCR on image
crop_text = pytesseract.image_to_string(final_img, config=config)
#text = pytesseract.image_to_string(new_image, config=config)
##text= pytesseract.image_to_string(Image.open(imagepath))
data = pytesseract.image_to_string(final_img)
# data is stored in CSV file
raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[data]}
df = pd.DataFrame(raw_data)
df.to_csv('data.csv',mode='a')

# print recognized text
print(data)

cv2.waitKey(0)
cv2.destroyAllWindows()
