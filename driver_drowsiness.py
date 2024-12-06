# Importing required libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to calculate the Eye Aspect Ratio (EAR)
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking eye state
    if ratio > 0.25:
        return 2  # Eyes open
    elif 0.21 < ratio <= 0.25:
        return 1  # Eyes partially closed
    else:
        return 0  # Eyes closed

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error: {e}\nEnsure 'shape_predictor_68_face_landmarks.dat' is in the script directory.")
    exit()

# Initialize counters and status
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Main loop for video capture
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Check eye states
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])

        # Update states
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                winsound.Beep(2500, 1000)  # Alert sound for sleeping
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                winsound.Beep(2000, 500)  # Alert sound for drowsy
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    # Display video frames
    cv2.putText(frame, "Press ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Drowsiness Detector", frame)

    # Exit on pressing ESC
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
