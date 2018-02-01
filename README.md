# ASL_Translation

**First time seeing this directory? Follow these 3 steps:**
1. Go to a directory that you're okay with putting these projects files in 
2. Type into terminal or command line: ```git clone https://github.com/Abhishaike/ASL_Translation.git```
3. Type into terminal or command line: ```cd ASL_Translation```

You'll have to repeat step 3 if you ever navigate to a different directory~


***************************************


**Want to save training data? Follow step 1 for hand-data, or step 2 for non-hand-data**
1. Type into terminal or command line: ```python3 data_collection.py -n YOUR_NAME -c CHARACTER_WANTED```. 

You'll be given a few seconds to prepare. Make sure you have some webcam-viewing application open during this ('Photobooth' for mac), so you can make sure that your hand is taking up a lot of the screen and is correctly positioned. Remember to move your hand angle around just *little* bit during the video process, diverse data is a good thing! However, never do any large angle changes, our feature extraction technique is NOT rotation-invariant. We will be using this: http://lifeprint.com/asl101/fingerspelling/images/signlanguageabc02.jpg as our ASL reference. Finally, always use your right hand when collecting data!


2. Type into terminal or command line: ```python3 data_collection.py -n YOUR_NAME -c None```

As with before, you'll be given a few seconds to prepare. And also as before, make sure you have some way to view your web-cam activites, there should be NO hands ever on the screen. Diverse data is also important here, move around a little and hold up objects (of course, without ever showing your actual hands on camera). 
