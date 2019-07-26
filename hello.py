#
# Import all the required libraries
#
import os
from keras.preprocessing.image import img_to_array

#
#   to remove directory forcefully
#

import shutil
from flask import Flask, redirect, url_for, request, flash, render_template
from werkzeug import secure_filename
from imutils import paths
import cv2
import random
import numpy as np
import keras
import zipfile
import tensorflow as tf

from flask_mysqldb import MySQL


#
#   Loading the saved model
#
model = keras.models.load_model("save_model.model")
graph = tf.get_default_graph()

app = Flask(__name__)

#
# Configuration for MySql
#

#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = 'aashishroot'
#app.config['MYSQL_DB'] = 'MyDB'

#mysql = MySQL(app)


ALLOWED_EXTENSIONS = set(['zip'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

label =[]
#
#   Display output on the next page
#
@app.route('/success')
def success():
    return render_template("Print.html", len= len(label), name=label)
    #return 'Its a  %s' % name



#
# Api call of /login
#
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if model:
        if request.method == 'POST':

            #
            #   Taking out the user name from the form data
            #
            user = request.form['name']

            #
            #   Taking out the file name from the form data
            #
            file = request.files['file']

            filename = secure_filename(file.filename)

            #
            #   This is the path where we storing the uploaded images
            #
            path = '/Users/zomato/Documents/Food_Rece/upload/'+user

            #
            #   If path is present then we first delete that path and again recreate that
            #
            if os.path.exists(path) == True:
                shutil.rmtree(path)

            #
            #   Creating the new path
            #
            os.makedirs(path)

            #
            #   This implementation is used in case of zip file
            #
            if file and allowed_file(filename):
                file.save(os.path.join(path,filename))  ## path is the upload folder
                zip_ref = zipfile.ZipFile(os.path.join(path, filename), 'r')
                zip_ref.extractall(path)
                zip_ref.close()

            path = path + '/single_prediction'

            file.save(os.path.join(path, filename))
            #var = predict_new.predicting(path)

            #
            #   retrieving the images from the image path
            #
            imagePaths = list(paths.list_images(path))
            random.shuffle(imagePaths)
            imagePaths = imagePaths[:]

            #
            #   initialising the list of results
            #
            results = []

            #label=list()

            #
            # Iterate over each image path and prediting the result
            #
            for p in imagePaths:
                #
                #   reading the image
                #
                original = cv2.imread(p)

                #
                #   converting the image from BGR to RGB
                #
                image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (64, 64))
                image = image.astype("float") / 255.0

                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)

                global graph
                with graph.as_default():
                    pred = model.predict(image)
                ##pred = pred.argmax(axis=1)[0]

                #
                # This is where we are labelling the output as Food and Non Food
                #

                label.append('Non_food' if pred > 0.5 else 'Food')   ### Dog and cat
                color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)

                #
                #   resize the orginal image and draw label with it
                #
                original = cv2.resize(original, (128, 128))

                #
                #   add output image to out list of images
                #

                #cur = mysql.connection.cursor()
                #cur.execute("INSERT INTO Photos(Path, Label) VALUES (%s,%s)", (p,label))
                #mysql.connection.commit()
                #cur.close()
                length = len(label)
                results.append(original)

            return redirect(url_for('success'))
    else:
        print ('Try training a model')


if __name__ == '__main__':
    app.run(use_reloader = True, debug = True)   ##app.run(host, port,debug,option) default : 127.0.0.1:5000

