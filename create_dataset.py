import mediapipe as mp
import cv2 
import os
import pickle

import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.8 )
DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                for  i in range(len(hand_lms.landmark)):
                    x = hand_lms.landmark[i].x
                    y = hand_lms.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data' : data, 'labels' : labels}, f)
f.close()


            

  