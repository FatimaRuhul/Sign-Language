import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape
        p1 =  0 , 0
        p2 = int (w*0.3), int(0.5 * h)
        # print(p1, p2)
        # print(h,w)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Please keep your hand inside the box.', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()