#
# Importing all the required libraries
#
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--images", required=True, help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

#
#   loading th pretrained model
#
print(" Loading the pre-trained network....")
model = load_model("save_model.model")

#
#   retrieving the images path from the image path
#

imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:]

#
#   initialising the list of results
#

results = []

#
# Iterating over each image and try to predict whether it is Food or non-food
#
for p in imagePaths:
    #
    #   Reading the images from the path
    #
    original = cv2.imread(p)

    #
    # Converting the image from BGR to RGB
    #
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64,64))
    image = image.astype("float") / 255.0   ## convert to float

    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    #
    #   Predicting the type here using the saved model
    #
    pred = model.predict(image)
    ##pred = pred.argmax(axis=1)[0]

    print(pred)

    label = "Non_food" if pred > 0.5 else "Food"
    color = (0,0,255) if pred > 0.5 else (0,255,0)

    #
    #   resize the orginal image and draw label with it
    #
    original = cv2.resize(original, (128,128))
    cv2.putText(original, label, (3,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #
    #   add output image to out list of images
    #
    results.append(original)


#
#   create a montage to display images with labels on it
#
montage = build_montages(results,(128,128),(14,7))[0]

#
#   show output montage
#
cv2.imshow("Results", montage)
cv2.waitKey(0)