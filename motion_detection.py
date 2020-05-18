# importing the libraries
import cv2

# reading the video 
cap = cv2.VideoCapture('cctv.mp4')

# storing the width and the height of the frame
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

# reading the first 2 frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

# processing the video
while cap.isOpened():
    
    # calculating the difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # converting the BGR to GRAY scale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # blurring the image 
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # threshing the image and dilating to find contours
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # modifying the contours by drawing a rectangle
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1500:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # resizing the frame
    image = cv2.resize(frame1, (1280,720))
    
    # displaying the frame
    cv2.imshow("feed", frame1)
    
    # modifying the frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # break condition `Esc`
    if cv2.waitKey(40) == 27:
        break

# freeing the resources
cv2.destroyAllWindows()
cap.release()
