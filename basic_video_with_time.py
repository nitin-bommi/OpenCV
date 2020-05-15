# importing the libraries
import cv2
import datetime

# rounding the time to nearest seconds
def roundSeconds(dateTimeObject):
    newDateTime = dateTimeObject

    if newDateTime.microsecond >= 500000:
        newDateTime = newDateTime + datetime.timedelta(seconds=1)

    return newDateTime.replace(microsecond=0)

# opening the camera
capture = cv2.VideoCapture(0)

print("Initial width: ",capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Initial height: ",capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

capture.set(3, 1280)
capture.set(4, 720)

print("Final width: ",capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Final height: ",capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writing the video to a file
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# processing the frame
while(capture.isOpened()):
    
    ret, frame = capture.read();
    
    if ret is True:
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        screen_resolution = 'Resolution: '+str(int(capture.get(3)))+' x '+str(int(capture.get(4)))
        current_time = 'Time: '+str(roundSeconds(datetime.datetime.now()))
        quit_key = 'Press q to exit'

        frame = cv2.putText(frame,
                            screen_resolution,
                            (10,50),
                            font,
                            0.75,
                            (0,0,0),
                            2,
                            cv2.LINE_AA)
    
        frame = cv2.putText(frame,
                            current_time,
                            (10,100),
                            font,
                            0.75,
                            (0,0,0),
                            2,
                            cv2.LINE_AA)
    
        frame = cv2.putText(frame,
                            quit_key,
                            (10,670),
                            font,
                            0.75,
                            (0,0,0),
                            2,
                            cv2.LINE_AA)
        
        output.write(frame)
    
        cv2.imshow('frame', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break
    
# freeing the resources
capture.release()
output.release()
cv2.destroyAllWindows()

