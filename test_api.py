import requests

# Here is your exact 3D file path!
file_path = r"C:\Users\Lenovo\OneDrive\Desktop\17012026-asad dk srdar-lowerjaw.stl"

print(f"Sending {file_path} to the AI Engine...")

try:
    with open(file_path, "rb") as f:
        # Send the file to our local server
        response = requests.post("http://127.0.0.1:8000/api/segment", files={"file": f})
    
    print("Status Code:", response.status_code)
    print("AI Response:", response.text)
except FileNotFoundError:
    print(f"ERROR: Could not find the 3D file at {file_path}. Please check the path!")