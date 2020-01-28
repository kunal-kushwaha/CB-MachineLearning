import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
count = 0
classifier = cv2.CascadeClassifier("haar.xml")

data = np.load("data.npy")

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier(3)

model.fit(X, y)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        for face in faces:
            x, y, w, h = face
            crop = gray[y:y+h, x:x+w]

            crop = cv2.resize(crop, (100, 100))
            flat = crop.flatten()

            ans = model.predict([flat])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, str(ans[0]), (x + 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imshow("Face Recog", frame)

        key = cv2.waitKey(1)

        if ord('q') == 0xff & key:
            break


cap.release()
cv2.destroyAllWindows()

