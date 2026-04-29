import os

while True:
    print("\n===== FACE ATTENDANCE DASHBOARD =====")
    print("1. Start Attendance")
    print("2. Train Model")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        os.system("py attendance.py")

    elif choice == "2":
        os.system("py train.py")

    elif choice == "3":
        print("Exiting...")
        break

    else:
        print("Invalid choice")