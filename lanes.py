import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    #Convert to grayscale image from color image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #Blur the image for reducing noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #Canny method to distinguish low and high intensity change in images
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

#Read the image
image = cv2.imread('test_image.jpg')

#Copy the image using numpy / duplicating so that the original image is not changed
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_image = region_of_interest(canny)

cv2.imshow("result", cropped_image)
cv2.waitKey(0)
#plt.imshow(canny)
#plt.show()
