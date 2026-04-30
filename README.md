# AI Face Recognition Attendance System

This is a modern web-based project built using Python, Streamlit, and OpenCV to detect and recognize faces and mark attendance automatically using a browser camera.

## What it does

- Detects faces using browser camera (no DroidCam required)
- Recognizes registered users using trained model
- Allows user login and signup
- Add and delete users easily
- Marks attendance with date and time
- Prevents duplicate attendance for the same day
- Displays attendance records and analytics

## Tech used

- Python
- OpenCV
- NumPy
- Pandas
- Streamlit
- PIL

## How to run

1. Signup / Login  
2. Add user (capture face images)  
3. Train the model  
4. Run Live Attendance  
5. View attendance records  

## Output

- Attendance is saved in `attendance.xlsx`
- Face data stored in `dataset/`
- Trained model saved as `model.yml`
