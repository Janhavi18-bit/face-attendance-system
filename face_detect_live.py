import cv2

cap = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    success, img = cap.read()
    
    if not success:
        print("Camera not working")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()