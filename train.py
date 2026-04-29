import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            faces.append(image)
            labels.append(current_id)

    current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("model.yml")

with open("labels.txt", "w") as f:
    for id, name in label_map.items():
        f.write(f"{id},{name}\n")

print("Training complete")