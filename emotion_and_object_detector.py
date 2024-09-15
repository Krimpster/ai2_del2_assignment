import keras
from keras.models import load_model
from keras.utils import img_to_array

import logging
import queue
import numpy as np
import cv2

from typing import List, NamedTuple

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import os
if not os.path.isfile("model.keras"):
    raise FileNotFoundError("model.keras file is missing!")

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_classifer = load_model(r"/mount/src/ai2_del2_assignment/model.keras")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

class VideoTransformer(VideoTransformerBase):
    def tranform_video(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                prediction = emotion_classifer.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x,y-7)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'Nothing to predict',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return img
    
def main():
    st.title("Real time facial emotion classifer")
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)