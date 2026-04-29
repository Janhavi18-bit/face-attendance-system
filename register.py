import cv2
import os

# ---------- USER INPUT ----------
name = input("Enter person name: ")

# ---------- CREATE FOLDER ----------
dataset_path = os.path.join("dataset", name)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# ---------- LOAD IMAGE ----------
img = cv2.imread("test.jpg")  # Put your image here

if img is None:
    print("❌ Image not found. Add test.jpg")
    exit()

# ---------- FACE DETECTOR ----------
cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ Haarcascade not loaded")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

count = 0

for (x, y, w, h) in faces:
    count += 1

    face = gray[y:y+h, x:x+w]

    file_path = os.path.join(dataset_path, f"{count}.jpg")
    cv2.imwrite(file_path, face)

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, f"Captured {count}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Register Face (Image Mode)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Face data collected from image!")

print("🔄 Auto training started...")

os.system("python auto_train.py")

print("✅ System updated automatically!")