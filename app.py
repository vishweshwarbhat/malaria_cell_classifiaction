
from flask import Flask,render_template,request,url_for
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


import os
#import sys
#import glob
import re

from werkzeug.utils import secure_filename
path="classify_model.h5"
model=load_model(path)


app=Flask(__name__)
@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def result():
    if request.method=="POST":
        f=request.files["file"]
        base_path=os.path.dirname(__file__)
        file_path=os.path.join(base_path,"uploads",secure_filename(f.filename))
        f.save(file_path)
        x=image.load_img(file_path,target_size=(224,224))
        x=image.img_to_array(x)
        image_data=np.expand_dims(x,axis=0)
        y_pred=model.predict(image_data)
        result=np.argmax(y_pred,axis=1)
        if result==0:
            pred="chances of infection is high \n please consult a doctor"
        if result==1:
            pred="you are safe"
        final=pred
        return final
    return None
if __name__ == '__main__':
    app.run(debug=True)