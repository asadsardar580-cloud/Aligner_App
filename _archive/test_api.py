import requests

# SCRUBBED 2026-09-16: this line carried a real patient-identifying
# filename (a date and a name). Replaced with a generic mock path.
# The original remains in git history from the baseline commit - see
# _archive/README.md for the options if that matters to you.
file_path = r"C:\path\to\patient_mock_mandible.stl"

print(f"Sending {file_path} to the AI Engine...")

try:
    with open(file_path, "rb") as f:
        # Send the file to our local server
        response = requests.post("http://127.0.0.1:8000/api/segment", files={"file": f})
    
    print("Status Code:", response.status_code)
    print("AI Response:", response.text)
except FileNotFoundError:
    print(f"ERROR: Could not find the 3D file at {file_path}. Please check the path!")