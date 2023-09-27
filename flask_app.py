import os
import flask
from flask import Flask, render_template, url_for, request, redirect
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('/Users/dipit/Image Data/COVIDX-Ray/model/XRayInceptionV3.h5')

@app.route("/",methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route("/", methods = ['GET', 'POST'])
def uploads():
    classes = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
    if request.method == "POST":
        f = request.files['imagefile']
        basepath = os.path.dirname(__file__)
        filename = f.filename
        image_path = os.path.join(
            basepath,'static/uploads', filename
        )
        f.save(image_path)
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = image/255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = classes[np.argmax(predictions[0])].lower()
        confidence = round(100*(np.max(predictions[0])),3)
        return render_template('index.html', filename = filename,prediction = predicted_class, 
                               confidence = confidence)
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
