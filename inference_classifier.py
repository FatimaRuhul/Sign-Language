import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3 )


cap = cv2.VideoCapture(0)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}


while True:

    data_aux =[]
    x_ = []
    y_ = []

    ret, frame = cap.read()

    h,w,c = frame.shape
    p1 =  0 , 0
    p2 = int (w*0.3), int(0.5 * h)
    cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        
    cv2.putText(frame, 'Please keep your hand inside the box.', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            #     frame,  # image to draw
            #     hand_landmarks,  # model output
            #     mp_hands.HAND_CONNECTIONS,  # hand connections
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())

        for hand_lms in results.multi_hand_landmarks:
                for  i in range(len(hand_lms.landmark)):
                    x = hand_lms.landmark[i].x
                    y = hand_lms.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)



        x1 = int(min(x_) * w)
        y1 = int(min(y_) * h)
        
        x2 = int(max(x_) * w)
        y2 = int(max(y_) * h)
        
        prediction = model.predict([np.asarray(data_aux)])
        
        predicted_char = labels_dict[int(prediction[0])]
        # print(predicted_char)

    

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0,), 4)
        cv2.putText(frame, predicted_char, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)


        
    cv2.imshow('Frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
