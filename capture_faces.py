import cv2
import os

cap = cv2.VideoCapture(0)

folder = "dataset/solan"

count = 0

while True:
    success, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capture", img)

    # Press ENTER to stop
    if cv2.waitKey(1) == 13:
        break

    # Save images
    if count < 30:
        cv2.imwrite(f"{folder}/{count}.jpg", gray)
        count += 1

cap.release()
cv2.destroyAllWindows()