# importing the libraries
import cv2

# storing the coordinates
points = []

# event function
def click_event(event, x, y, flag, params):
    
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 2, (0,0,0), -1)
        points.append((x,y))
        
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (0,0,0), 2)
            points = []
        
        cv2.imshow('Scientists', img)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        coordinates = 'Coordinates: ('+str(x)+','+str(y)+')'
        cv2.putText(img, coordinates, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0))
        cv2.imshow('Scientists', img)
        
# reading the image
img = cv2.imread('scientists.jpg', -1)

# displaying the image
cv2.imshow('Scientists', img)

# event listener method
cv2.setMouseCallback('Scientists', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
