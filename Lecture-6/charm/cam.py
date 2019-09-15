import cv2
import numpy as np
cap = cv2.VideoCapture(0)

count = 0

# make classifier to detect face
classifier = cv2.CascadeClassifier("haar.xml")

while True:
    ret, frame = cap.read()
    # this will return a boolean(if we got image or not) and the images numpy array


    if ret:
        # cv2.imshow("Window", frame)

        # this will take it image in gray color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)
        # print(faces)    # each face will have [x,y,width,height]
        # x means when you got face when looking from left to right, w means how long did face go on x axis

        if len(faces) > 0:

            arr = np.array(faces)[:, 2:]
            index = np.prod(arr, axis=1).argmax()

            face = faces[index]
            x, y, w, h = face
            crop = gray[y:y+h, x:x+w]   # notice : starts from x and ends at x+w means width of face is w
            crop = cv2.resize(crop, (200, 200))
            cv2.imshow("Crop", crop)

        # crop = gray[:300, :300]
        cv2.imshow("Window", gray)
        # cv2.imshow("Crop", crop)

    # key = cv2.waitKey(1000)
    # waitKey acts as waiting for 1 wait time (refresh rate) and also returns the value of key you pressed
    key = cv2.waitKey(1)
    if ord('q') == 0xff & key:  # & key means it will give same number but now it will be in 8 bits as 0xff = 8 bits
        break

    if ord('c') == 0xff & key:
        cv2.imwrite(f"{count}.png", frame)
        count += 1

cap.release()   # stops camera
cv2.destroyAllWindows()