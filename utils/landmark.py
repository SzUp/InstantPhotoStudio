import numpy as np
import dlib
from PIL import Image, ImageDraw
from .alignment import test
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def get_68landmarks_img(img,model):
    img = test(img)
    model = test(model)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)
    landmarks = []
    x_l = []
    y_l = []
    for face in faces:
        shape = predictor(gray, face)
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            if(i<48):
                landmarks.append((x + 96, y + 128))
            if(i>=48):
                x_l.append(x)
                y_l.append(y)

    x_ori = np.array(x_l).mean()
    y_ori = np.array(y_l).mean()
    
    gray = cv2.cvtColor(model, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)
    x_l = []
    y_l = []
    x_targ = []
    y_targ = []
    for face in faces:
        shape = predictor(gray, face)
        for i in range(68):
            if(i>=48):
                x = shape.part(i).x
                y = shape.part(i).y                
                landmarks.append((x, y))
                x_l.append(x)
                y_l.append(y)
    
    x_targ = np.array(x_l).mean()
    y_targ = np.array(y_l).mean()

    x_diff = x_ori - x_targ
    y_diff = y_ori - y_targ

    # merge
    for i in range(48,68):
        landmarks[i] = (landmarks[i][0] + x_diff + 96 , landmarks[i][1] + y_diff + 128)

    con_img = Image.new('RGB', (384, 384), color=(0, 0, 0))
    # con_img = Image.new('RGB', (img.shape[1], img.shape[0]), color=(0, 0, 0))
    draw_landmarks(con_img, landmarks)
    # con_img = con_img.resize((512, 512))
    con_img = np.array(con_img)
    return con_img