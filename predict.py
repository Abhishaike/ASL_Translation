import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

def preprocess(image):
    open_cv_image = np.array(image)
    blur = cv2.GaussianBlur(open_cv_image,(5,5),2)   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, processed_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return processed_image

font = cv2.FONT_HERSHEY_SIMPLEX
topRight = (10, 100)
fontScale = 0.9
fontColor = (255, 255, 255)
lineType = 2

model = load_model('24class_model.h5')

img_height = 100
img_width = 100  

y = 100
x = 100
h = 100
w = 100
 
textStartingY = 100

letter_map = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

cap = cv2.VideoCapture(0)

while(True):
    
    # Capture frame-by-frame
    cat, frame = cap.read()

    frameResized = frame[y:y+h, x:x+w]

    frameResized = cv2.resize(frameResized, (img_height, img_width))
    # Our operations on the frame come here

    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)

    #Converting to black white like the training images
    frameResized = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
    frameResized = preprocess(frameResized)

    cv2.imshow('frameResized',frameResized)

    #reshaping for nn
    frameResized = frameResized.reshape(1, img_height, img_width,1)

    result = model.predict(frameResized)[0]  # Predict
    guessed_letter = letter_map[np.argmax(result)]
    for i in range(0,len(result)):
        strToPrint0 = str(letter_map[i]) +': ' + str(round(result[i],2))
        if (i < 12): # Adds second row of 12 characters starting back at the top
            cv2.putText(frame, strToPrint0, (10, textStartingY + i * 30), font, fontScale, fontColor, lineType)
        else:
            cv2.putText(frame, strToPrint0, (300, textStartingY + (i - 12) * 30), font, fontScale, fontColor, lineType)
    cv2.putText(frame, guessed_letter, (220, textStartingY), font, fontScale, fontColor, lineType)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

