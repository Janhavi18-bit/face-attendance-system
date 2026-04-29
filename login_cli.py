USERNAME = "admin"
PASSWORD = "admin123"

u = input("Username: ")
p = input("Password: ")

if u == USERNAME and p == PASSWORD:
    print("Login successful")
    import os
    os.system("python dashboard.py")
else:
    print("Invalid credentials")