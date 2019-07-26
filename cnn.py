#This file is mainly responsible for saving the trained model
#First we are creating our CNN model
#Second we are saving the model to the save_model.model
#then we plotting the graph of the trained model for diffrent parameter i.e. accuracy, loss ete


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())


# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#
# This command will take out the training data from the directory i.e dataset/training_set
#
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
#
# This command will take out the testing data from the directory i.e dataset/test_set
#
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

H = classifier.fit_generator(training_set,
                        steps_per_epoch=3000 // 32,  ## steps_per_epoch = sample_per_epoch/batch size
                        epochs=10,
                        validation_data=test_set,
                        validation_steps=500 // 32)




#
# to save the netwok to the disk
#
print("[INFO] seriralizinf network to '{}'...".format("save_model.model"))
classifier.save("save_model.model")

#
#plot
#

N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy on dataser")
plt.xlabel("Epoch #")
plt.ylabel("Loss/accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")

##

# Part 3 - Making new predictions


#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/50.jpg', target_size=(64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices

#print(result)

#if result[0][0] == 1:
#    prediction = 'Its a dog'
#else:
#    prediction = 'Its a cat'

#print(prediction)

#test_image = image.load_img('dataset/single_prediction/50.jpg', target_size=(64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices

#print(result)

#if result[0][0] == 1:
#    prediction = 'Its a dog'
#else:
#    prediction = 'Its a cat'

#print(prediction)
