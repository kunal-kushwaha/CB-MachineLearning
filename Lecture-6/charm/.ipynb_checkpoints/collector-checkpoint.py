import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
classifier = cv2.CascadeClassifier("haar.xml")

name = input("Enter your name : ")

images = []

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        if len(faces):
            index = np.array(faces)[:, 2:].prod(axis=1).argmax()
            face = faces[index]
            x, y, w, h = face
            crop = gray[y:y+h, x:x+w]

            crop = cv2.resize(crop, (100, 100))
            cv2.imshow("Crop", crop)

        cv2.imshow("Window", frame)
        key = cv2.waitKey(1)

        if ord('q') == 0xff & key:
            break

        if ord('c') == 0xff & key:
            if len(faces):
                images.append(crop.flatten())
            count += 1
            print("Captured : " + str(count))
            if count >= 10:
                break


X = np.array(images)
y = np.full((X.shape[0], 1), name)

data = np.hstack([y, X])

try:
    old = np.load("data.npy")
except FileNotFoundError as e:
    old = np.zeros([0, 10001], dtype=int)

result = np.vstack([old, data])

np.save("data.npy", result)

print(result.shape)

cap.release()
cv2.destroyAllWindows()
