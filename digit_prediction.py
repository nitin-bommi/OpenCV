import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')

dataset = fetch_openml('mnist_784', version=1)
X, y = dataset['data'], dataset['target'].astype(np.uint8)

knn_classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_classifier.fit(X, y)

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)        


img = np.zeros((200,200), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

img = Image.fromarray(img)
foo = img.resize((28,28),Image.ANTIALIAS)
foo = np.array(foo)
plot_digit(foo)
flat = foo.flatten()
zero = X[0]
knn_classifier.predict([flat])
