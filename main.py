import asyncio
from typing import List, Tuple
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import cv2, pickle
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import numpy as np
import mediapipe as mp

# app =FastAPI() # initialize FastAPI
# initialize the classifier that we will use
model = load_model(r"model.h5")
mpHands = mp.solutions.hands
hands = mpHands.Hands()


def model_prediction(model, landmarks):
    pred_prob = model.predict(landmarks)
    pred_class = np.argmax(pred_prob)
    return pred_class, max(max(pred_prob))


def process_image(frame):
    # frame = frame.to_ndarray(format="bgr24")
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frameRGB

def get_landmarks(result):
    lmsList = []
    # result = hands.process(frameRGB)
    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        for lm in handLms.landmark:
            # h, w, c = frameRGB.shape
            lmsList.append(lm.x)
            lmsList.append(lm.y)
            lmsList.append(lm.z)
        lmsList = [lmsList]
        lmsList = np.array(lmsList)
    return lmsList


def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def get_text_from_database(pred_class):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    command = "SELECT label FROM gesture WHERE label_index=" + str(pred_class)
    cursor.execute(command)
    for row in cursor:
        return row[0]

def get_pred_from_landmarks(lms):
    text = ""
    pred_class, pred_prob = model_prediction(model, lms)
    # print(pred_class, pred_prob)
    if (pred_prob * 100 > 60):
        text = get_text_from_database(pred_class)
    return text


is_voice_on = True



@asynccontextmanager
async def lifespan(app: FastAPI):
    # print("Execute before closing")
    
    
    # cascade_classifier.load(
    #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # )
    yield
    # print("Execute before closing")

app = FastAPI(lifespan=lifespan)





class Faces(BaseModel):
# """ This is a pydantic model to define the structure of the streaming data 
# that we will be sending the the cv2 Classifier to make prediction
# It expects a List of a Tuple of 4 integers
# """
    faces: List[str]

async def receive(websocket: WebSocket, queue: asyncio.Queue):
# """
# This is the asynchronous function that will be used to receive webscoket 
# connections from the web page
# """
    bytes = await websocket.receive_bytes()
    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
# """
# This function takes the received request and sends it to our classifier
# which then goes through the data to detect the presence of a human face
# and returns the location of the face from the continous stream of visual data as a
# list of Tuple of 4 integers that will represent the 4 Sides of a rectangle
# """
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        bytes = await queue.get()
        data = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        frameRGB = process_image(img)
        result = hands.process(frameRGB)
        old_text = text
        if result.multi_hand_landmarks:
            lms = get_landmarks(result)
            # print(lms)
            text = get_pred_from_landmarks(lms)
            if(old_text == text):
                count_same_frame += 1
            else:
                count_same_frame = 0
                
            if count_same_frame > 30:
                word = word + text
                count_same_frame = 0
                print((word + "  ") * 50)
                faces_output = Faces(faces=[word])
        else:
            faces_output = Faces(faces=[])

        await websocket.send_json(faces_output.dict())


@app.websocket("/face-detection")
async def face_detection(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=10)
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except WebSocketDisconnect:
        detect_task.cancel()
        await websocket.close()
# @app.on_event("startup")
# async def startup():
