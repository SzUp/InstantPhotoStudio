
import dlib
import face_recognition
import math
import numpy as np
import cv2

def rect_to_bbox(rect):
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)
 
 
def face_alignment(faces):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy, dx) * 180. / math.pi

        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned



def test(img_path):
    unknown_image = face_recognition.load_image_file(img_path)

    face_locations = face_recognition.face_locations(unknown_image)

    src_faces = []
    src_face_num = 0
    for (i, rect) in enumerate(face_locations):
        src_face_num = src_face_num + 1
        (x, y, w, h) = rect_to_bbox(rect)
        detect_face = unknown_image[y:y+h, x:x+w]
        src_faces.append(detect_face)
        detect_face = cv2.cvtColor(detect_face, cv2.COLOR_RGBA2BGR)
    faces_aligned = face_alignment(src_faces)
    face_num = 0
    for faces in faces_aligned:
        face_num = face_num + 1
        faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
        return faces
