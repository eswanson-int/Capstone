import streamlit as st
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd


st.title("Webcam Emotion Detector")
st.write("This is a streamlit App where it looks at live webcam feed and"
         "tells you over time if you are enjoying the activity or not")

# Load the cascade for face detection and models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# access the facial expression class where the image is classified
#model = ExpressionModel("model.json", "model_weights.h5")
global emotions
emotions = ["Angry", "Disgust", "Fear", "Happy",
            "Neutral", "Sad", "Surprise"]

# load model from JSON file
global loaded_model

@st.cache(allow_output_mutation=True)
def load_data():
    with open('model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model_weights.h5')
    return loaded_model

#def load_data(model_file, model_weights_file):
#    with open(model_file, "r") as json_file:
#        loaded_model_json = json_file.read()
#        loaded_model = model_from_json(loaded_model_json)
#        loaded_model.load_weights(model_weights_file)
#    return loaded_model
#model = load_data('model.json', 'model_weights.h5')

model = load_data()

def predict_emotion(img):
    #model = load_data('model.json', 'model_weights.h5')
    preds = model.predict(img)

    return emotions[np.argmax(preds)]



#define text font
font = cv2.FONT_HERSHEY_SIMPLEX

global pred2
global happy
global sad
#test = 0
happy = 0
sad = 0

#create the dataFrame to store information
d = {'Sad Time' : [0], 'Happy Time': [0]}
df = pd.DataFrame(data = d)
#df.set_index('sad', inplace=True)


run = st.checkbox('Run')
window = st.image([])

# Access video from webcam.
cap = cv2.VideoCapture(0)

H = st.empty()
S = st.empty()

while run:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        face_capture = gray_img[y:y + h, x:x + w]
        #resize image for the model
        resize = cv2.resize(face_capture, (48, 48))
        #make prediction of image from image classifier model
        pred = predict_emotion(resize[np.newaxis, :, :, np.newaxis])



        #Text from classifier on image
        font_scale = 1
        line_type = 2
        cv2.putText(img, pred, (x, y), font, font_scale, (255, 255, 0), line_type)
        #Rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #save the information in a dataframe table
        if pred == 'Happy':
            happy += 1
        if pred == 'Sad':
            sad += 1

        with H.empty():
            H.write(f'the happy count is {happy}')

        with S.empty():
            S.write(f'the sad count is {sad}')

        #@st.cache
        def write_temp(sad, happy):
            df['Sad Time'] = sad
            df['Happy Time'] = happy

            df['results'] = np.where(df['Happy Time'] > df['Sad Time'],
                                     "Looks like you really enjoyed that", "That wasn't fun, try something else")

            df.to_csv('temp.csv', index=False)

        write_temp(sad, happy)

    # Display
    #This method opens an outside window.  cv2 can't be used in streamlit
    #cv2.imshow('img', img)

    #To use inside streamlit - need to change coloring back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window.image(img)


# Stop if escape key is pressed
k = cv2.waitKey(30) & 0xff
if k==27:
    cap.release()()


clicked = st.button("View results")
if clicked:
    df = pd.read_csv('temp.csv')
    #df.set_index('happy', inplace=True)
    st.table(df)


