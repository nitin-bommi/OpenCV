# importing the libraries
import numpy as np
import cv2

# capturing the video
cap = cv2.VideoCapture(0)

# defining the onChange function for track bars
def scroll(x):
    pass

# creating a named window for track bars
cv2.namedWindow('Tracking')

cv2.createTrackbar('LH', 'Tracking', 0, 255, scroll)
cv2.createTrackbar('LS', 'Tracking', 0, 255, scroll)
cv2.createTrackbar('LV', 'Tracking', 0, 255, scroll)
cv2.createTrackbar('UH', 'Tracking', 255, 255, scroll)
cv2.createTrackbar('US', 'Tracking', 255, 255, scroll)
cv2.createTrackbar('UV', 'Tracking', 255, 255, scroll)

# procesisng the video 
while True:
    
    # capturing the frame
    _, frame = cap.read()
    
    # converting the frame from BGR to HSV format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # retrieving the track bar values
    l_h = cv2.getTrackbarPos('LH', 'Tracking')
    l_s = cv2.getTrackbarPos('LS', 'Tracking')
    l_v = cv2.getTrackbarPos('LV', 'Tracking')
    
    u_h = cv2.getTrackbarPos('UH', 'Tracking')
    u_s = cv2.getTrackbarPos('US', 'Tracking')
    u_v = cv2.getTrackbarPos('UV', 'Tracking')
    
    # defining the lower and upper bounds for the retreiving color
    l_b = np.array([l_h,l_s,l_v])
    u_b = np.array([u_h,u_s,u_v])
    
    # thresholding the hsv image to get the required color
    mask = cv2.inRange(hsv, l_b, u_b)
    
    # using bitwise operation to detect the color
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
