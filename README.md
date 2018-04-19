# C-ASL (Convolutional American Sign Language)
![demo](https://user-images.githubusercontent.com/20051469/38962344-7511cd4c-4332-11e8-8666-f1485ca24ee7.gif)

**First time seeing this directory? Follow these 3 steps:**
1. Go to a directory that you're okay with putting these projects files in 
2. Type into terminal or command line: ```git clone https://github.com/Abhishaike/ASL_Translation.git```
3. Type into terminal or command line: ```cd ASL_Translation```

You'll have to repeat step 3 if you ever navigate to a different directory~

***************************************

# Run the model

Run predict.py. This will open up your webcam and draw a small blue box on the screen. Place the palm of your hand so that the middle of your palm covers the box, then press space. Then, the box should follow your hand around and the translations of your hand symbols should appear in white at the top left of the screen. Translations are better if you keep your face out of the box and do it on a neutral background.
Needs cv2, numpy, keras and PIL installed.

***************************************

# Train

First download the data from.
Link to the Alex's data and built 24 class model  - https://drive.google.com/drive/folders/1VlhTLQ6MeMyVh-sVBBcnKz46i6Tx9ODn	

**Want to create new training data? Follow step 1 for hand-data, or step 2 for non-hand-data**
1. Type into terminal or command line: ```python3 data_collection.py -n YOUR_NAME -c CHARACTER_WANTED```. 

You'll be given a few seconds to prepare. Make sure you have some webcam-viewing application open during this ('Photobooth' for mac), so you can make sure that your hand is taking up a lot of the screen and is correctly positioned. Remember to move your hand angle around just *little* bit during the video process, diverse data is a good thing! However, never do any large angle changes, our feature extraction technique is NOT rotation-invariant. We will be using this: http://lifeprint.com/asl101/fingerspelling/images/signlanguageabc02.jpg as our ASL reference. Finally, always use your right hand when collecting data!


2. Type into terminal or command line: ```python3 data_collection.py -n YOUR_NAME -c None```

As with before, you'll be given a few seconds to prepare. And also as before, make sure you have some way to view your web-cam activites, there should be NO hands ever on the screen. Diverse data is also important here, move around a little and hold up objects (of course, without ever showing your actual hands on camera). 

**Training a new model**

Download https://drive.google.com/file/d/1PEAWb_f86C8IZEkCujE7fohXrDaNVf0c/view?usp=sharing and extract it to the repository.
Run 26class.py. This will train the model from the data in the ```\data-one\``` directory along with the kaggle dataset. It separates training and testing data automatically, but within the data file there must be a different file for each class. The new model will be saved as model.h5, and will overwrite any model files you already have in the same directory.
