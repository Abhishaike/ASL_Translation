import cv2
from keras.models import load_model
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
topRight = (10, 100)
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2

model = load_model('keras_model.h5')

img_height = 64
img_width = 64

y = 100
x = 100
h = 200
w = 200

cap = cv2.VideoCapture(0)



while(True):
    # Capture frame-by-frame
    cat, frame = cap.read()

    frameResized = crop_img = frame[y:y+h, x:x+w]

    frameResized = cv2.resize(frameResized, (img_height, img_width))
    # Our operations on the frame come here

    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)

    cv2.imshow('frameResized',frameResized)

    frameResized = np.expand_dims(frameResized, axis=0)
    # frameResized.shape # (1,64,64,3)


    result = model.predict(frameResized)[0]  # Predict

    print(result)
    prediction = result.tolist().index(max(result))  # The index represents the number predicted in this case

    strToPrint0 = 'A: ' + str(round(result[0],3))
    strToPrint1 = 'B: ' + str(round(result[1], 3))
    strToPrint2 = 'C: ' + str(round(result[2], 3))
    strToPrint3 = 'Five: ' + str(round(result[3], 3))
    strToPrint4 = 'Point: ' + str(round(result[4], 3))
    strToPrint5 = 'V: ' + str(round(result[5], 3))

    #if(prediction == 0):    guess = 'Guess: A'
    #else:   guess = 'Guess: B'
    guess = str(prediction)
    cv2.putText(frame, strToPrint0, topRight, font, fontScale, fontColor, lineType)
    cv2.putText(frame, strToPrint1, (10, 130), font, fontScale, fontColor, lineType)
    cv2.putText(frame, strToPrint2, (10, 160), font, fontScale, fontColor, lineType)
    cv2.putText(frame, strToPrint3, (10, 190), font, fontScale, fontColor, lineType)
    cv2.putText(frame, strToPrint4, (10, 220), font, fontScale, fontColor, lineType)
    cv2.putText(frame, strToPrint5, (10, 250), font, fontScale, fontColor, lineType)
    cv2.putText(frame, guess, (10, 280), font, fontScale, fontColor, lineType)



    # Display the resulting frame
    cv2.imshow('frame',frame)

    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# score = model.evaluate(validation_generator, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()



