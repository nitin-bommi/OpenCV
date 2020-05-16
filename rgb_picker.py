# importing the libraries
import numpy as np
import cv2

def scroll(x):
    print(x)

# img = np.zeros((720,1280,3), dtype=np.uint8)

cv2.namedWindow('Color picker')

cv2.createTrackbar('Blue', 'Color picker', 0, 255, scroll)
cv2.createTrackbar('Green', 'Color picker', 0, 255, scroll)
cv2.createTrackbar('Red', 'Color picker', 0, 255, scroll)

cv2.createTrackbar('Switch', 'Color picker', 0, 1, scroll)

while(True):
    
    img = np.zeros((720,1280,3), dtype=np.uint8)
    
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    b = cv2.getTrackbarPos('Blue', 'Color picker')
    g = cv2.getTrackbarPos('Green', 'Color picker')
    r = cv2.getTrackbarPos('Red', 'Color picker')
    s = cv2.getTrackbarPos('switch', 'Color picker')
    
    text = '('+str(r)+','+str(b)+','+str(g)+')'
    
    cv2.putText(img, text, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (abs(255-b),abs(255-g),abs(255-r)), 4)

    if s == 0:
       img[:] = 0
    else:
       img[:] = [b, g, r]
       
       
    cv2.imshow('Color picker',img)
    
cv2.destroyAllWindows()
