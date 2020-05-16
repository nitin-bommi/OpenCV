# importing the libraries
import numpy as np
import cv2

# defining onChange function for TrackBar
def scroll(x):
    print(x)

cv2.namedWindow('Color picker')

# creating the track bars
cv2.createTrackbar('Blue', 'Color picker', 0, 255, scroll)
cv2.createTrackbar('Green', 'Color picker', 0, 255, scroll)
cv2.createTrackbar('Red', 'Color picker', 0, 255, scroll)
cv2.createTrackbar('switch', 'Color picker', 0, 1, scroll)

while(True):
    
    # defining the image
    img = np.zeros((720,1280,3), dtype=np.uint8)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # retrieving the track bar position
    b = cv2.getTrackbarPos('Blue', 'Color picker')
    g = cv2.getTrackbarPos('Green', 'Color picker')
    r = cv2.getTrackbarPos('Red', 'Color picker')
    s = cv2.getTrackbarPos('switch', 'Color picker')

    # changing the color only if the switch is ON(1)
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]
       
    # displaying the image
    cv2.imshow('Color picker',img)
    
cv2.destroyAllWindows()
