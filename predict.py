import cv2
from keras.models import load_model
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
topRight = (10, 100)
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2

model = load_model('abc_binary_model.h5')

img_height = 200
img_width = 200      

y = 100
x = 100
h = 200
w = 200

textStartingY = 100
minValue = 0 #for black-white conversion

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    cat, frame = cap.read()

    frameResized = frame[y:y+h, x:x+w]

    frameResized = cv2.resize(frameResized, (img_height, img_width))
    # Our operations on the frame come here

    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)

    #Converting to black white like the training images
    gray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    ret, frameResized = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow('frameResized',frameResized)

    #reshaping for nn
    frameResized = frameResized.reshape(1, img_height, img_width,1)

    result = model.predict(frameResized)[0]  # Predict

    for i in range(0,len(result)):
        strToPrint0 = str(i) +': ' + str(round(result[i],3))
        cv2.putText(frame, strToPrint0, (10, textStartingY + i * 30), font, fontScale, fontColor, lineType)        

    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

