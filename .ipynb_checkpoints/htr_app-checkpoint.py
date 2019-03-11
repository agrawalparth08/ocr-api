from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
from PIL import Image
import numpy as np
import flask
import io
import cv2
import numpy as np
from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
# initialize our Flask application and the Keras model

photos = UploadSet('photos', IMAGES)
app = flask.Flask(__name__)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/'
configure_uploads(app, photos)


model = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    model = loaded_model
    model._make_predict_function()

def ocr_prediction(filename):
    print(filename)
    characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
	
    image = cv2.imread("static/"+filename)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    thresh = 127
    im_bw = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)[1]

    height,width = im_bw.shape
    im_bw = cv2.resize(im_bw,dsize = (width*5,height*4),interpolation = cv2.INTER_CUBIC)

    
    ret,thresh = cv2.threshold(im_bw,127,255,cv2.THRESH_BINARY_INV)

    im2,ctrs,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    m = list()
    sorted_ctrs = sorted(ctrs, key = lambda ctr: cv2.boundingRect(ctr)[0])
    pchl = list()

    dp = im_bw.copy()

    response = []
    for i,ctr in enumerate(sorted_ctrs):
        x,y,w,h = cv2.boundingRect(ctr)
    
        roi = im_bw[y-10:y+h+10, x-10:x+w+10]
    
        roi = cv2.resize(roi,dsize = (28,28), interpolation = cv2.INTER_AREA)
        kernel = np.ones((2,2),np.uint8)
        roi = cv2.erode(roi,kernel,iterations = 1)
        

        roi = np.array(roi)
        t = np.copy(roi)
        t = t /255.0
        t = 1 - t
        t = t.reshape(1,784)
        

        prob = model.predict_proba(t)
        prob_list = prob[0].argsort()[-3:][::-1]
     


        # top_response = {"t1":[characters[prob_list[0]],prob[0][prob_list[0]]],
        # "t2":[characters[prob_list[1]],prob[0][prob_list[1]]],
        # "t3":[characters[prob_list[2]],prob[0][prob_list[2]]]}
        # print()
        # print("TOP:",top_response)


        response.append(characters[prob_list[0]])

    return response
  


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST" and 'photo' in request.files:
            filename = photos.save(request.files['photo'])
            data["predictions"] = ocr_prediction(filename)
            print(data["predictions"])

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host="0.0.0.0", port=10000)