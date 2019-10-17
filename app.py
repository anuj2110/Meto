from for_server import *
from flask import Flask,request,jsonify
import tensorflow as tf
from PIL import Image
import io
import numpy as np 
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import cv2

model = get_full_model()
model.load_weights("bmi_model_weights.h5")
graph = tf.get_default_graph()
cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    img1 = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces_rects = cas.detectMultiScale(img1, scaleFactor = 1.2, minNeighbors = 5)
    x,y,w,h = faces_rects[0]
    img_h, img_w = np.shape(img1)
    xw1 = max(int(x - 0.1 * w), 0)
    yw1 = max(int(y - 0.1 * h), 0)
    xw2 = min(int(x+w + 0.1 * w), img_w - 1)
    yw2 = min(int(y+h + 0.1 * h), img_h - 1)
    face = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (224,224)) / 255.00
    face = np.expand_dims(face, axis=0)
    # return the processed image
    return face
app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def predict():
    data = {"success":False}
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
            data["prediction"]=[]
            global graph
            with graph.as_default():
                predictions = model.predict(image)
                r = {"result":float(predictions[0][0])}
                data["prediction"].append(r)
                data["success"] = True
    return jsonify(data)
if __name__=="__main__":
    app.run(debug=True)
