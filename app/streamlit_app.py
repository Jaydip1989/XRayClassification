import streamlit as st
from keras.models import load_model
from PIL import Image


from util import classify, set_background


set_background('png-bg/bg.png')

# set title
st.title('Pneumonia classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model/XRayInceptionV3.h5')

# load class names
classes = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    predicted_class, confidence = classify(image, model, classes)

    # write classification
    st.write("## {}".format(predicted_class))
    st.write("### score: {}%".format(int(confidence * 10) / 10))