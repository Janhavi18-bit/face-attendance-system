import cv2
import numpy as np

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

labels = {0: "solan"}  # your name mapping

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        id, confidence = recognizer.predict(face)

        name = labels.get(id, "Unknown")

        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Recognition", img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()