import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import hashlib

# ================= CONFIG =================
st.set_page_config(page_title="Face Attendance System")

DATASET = "dataset"
MODEL = "model.yml"
LABELS_FILE = "labels.txt"
ATT_FILE = "attendance.xlsx"
USER_FILE = "users.txt"

os.makedirs(DATASET, exist_ok=True)

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False

# ================= AUTH =================
def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def signup():
    st.title("Signup")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Create Account"):
        if user and pwd:
            with open(USER_FILE, "a", encoding="utf-8") as f:
                f.write(f"{user.strip()},{hash_pwd(pwd)}\n")
            st.success("Account created")
        else:
            st.warning("Fill all fields")

    st.stop()

def login():
    st.title("Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if os.path.exists(USER_FILE):
            with open(USER_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    u, p = parts
                    if u == user and p == hash_pwd(pwd):
                        st.session_state.login = True
                        st.rerun()
                        return

        st.error("Invalid credentials")

    st.stop()

# ================= ATTENDANCE FUNCTION (IMPORTANT POSITION) =================
def mark_attendance(name):
    if not os.path.exists(ATT_FILE):
        df = pd.DataFrame(columns=["Name", "Time", "Date"])
        df.to_excel(ATT_FILE, index=False)

    df = pd.read_excel(ATT_FILE)

    # ensure columns exist
    for col in ["Name", "Time", "Date"]:
        if col not in df.columns:
            df[col] = ""

    today = datetime.now().strftime("%Y-%m-%d")

    if not df[(df["Name"] == name) & (df["Date"] == today)].empty:
        return False

    df.loc[len(df)] = [
        name,
        datetime.now().strftime("%H:%M:%S"),
        today
    ]

    df.to_excel(ATT_FILE, index=False)
    return True

# ================= AUTH FLOW =================
auth = st.sidebar.selectbox("Auth", ["Login", "Signup"], key="auth")

if auth == "Signup":
    signup()

if not st.session_state.login:
    login()

# ================= MENU =================
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Add User", "Delete User", "Train Model", "Live Attendance", "View Attendance"],
    key="menu"
)

# ================= HOME =================
if menu == "Home":
    st.title("Face Attendance System")
    st.write("System ready 🚀")

# ================= ADD USER =================
elif menu == "Add User":
    st.title("Add User")

    name = st.text_input("Enter Name")
    img = st.camera_input("Capture Face")

    if img and name:
        img_pil = Image.open(img)
        img_np = np.array(img_pil)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        path = os.path.join(DATASET, name)
        os.makedirs(path, exist_ok=True)

        count = len(os.listdir(path))
        cv2.imwrite(f"{path}/{count}.jpg", gray)

        st.success("Image saved")

# ================= DELETE USER =================
elif menu == "Delete User":
    st.title("Delete User")

    users = os.listdir(DATASET)

    if users:
        user = st.selectbox("Select User", users, key="delete_user")

        if st.button("Delete"):
            path = os.path.join(DATASET, user)

            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

            os.rmdir(path)

            st.success("Deleted")
            st.rerun()
    else:
        st.warning("No users found")

# ================= TRAIN MODEL =================
elif menu == "Train Model":
    st.title("Train Model")

    if st.button("Train"):

        faces = []
        labels = []
        label_map = {}
        current_id = 0

        for person in os.listdir(DATASET):
            person_path = os.path.join(DATASET, person)

            if person not in label_map:
                label_map[person] = current_id
                current_id += 1

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)

                img = Image.open(img_path).convert("L")
                img_np = np.array(img)

                faces.append(img_np)
                labels.append(label_map[person])

        if len(faces) == 0:
            st.error("No data to train")
            st.stop()

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL)

        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            for name, id in label_map.items():
                f.write(f"{id},{name}\n")

        st.success("Model trained")

# ================= LIVE ATTENDANCE =================
elif menu == "Live Attendance":
    st.title("Live Attendance")

    if not os.path.exists(MODEL):
        st.error("Train model first")
        st.stop()

    img = st.camera_input("Capture Face")

    if img:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL)

        labels = {}
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    id, name = parts
                    labels[int(id)] = name

        img_pil = Image.open(img)
        img_np = np.array(img_pil)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        try:
            id, conf = recognizer.predict(gray)
            name = labels.get(id, "Unknown")
        except:
            name = "Unknown"
            conf = 100

        if conf < 70 and name != "Unknown":
            if mark_attendance(name):
                st.success(f"Marked: {name}")
            else:
                st.warning("Already marked today")
        else:
            st.error("Unknown face")

# ================= VIEW =================
elif menu == "View Attendance":
    st.title("Attendance Records")

    if os.path.exists(ATT_FILE):
        df = pd.read_excel(ATT_FILE)
        st.dataframe(df)

        if not df.empty:
            st.bar_chart(df["Name"].value_counts())
    else:
        st.warning("No data")