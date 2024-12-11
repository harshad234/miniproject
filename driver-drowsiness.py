# Importing required libraries
import cv2
import numpy as np
import dlib
import winsound
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import face_utils  # Ensure imutils is installed

# Verify required files are present
eye_model_path = "eye_state_model.h5"
landmark_model_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(eye_model_path):
    print(f"Error: '{eye_model_path}' not found. Please ensure the eye state model file is in the script directory.")
    exit()

if not os.path.exists(landmark_model_path):
    print(f"Error: '{landmark_model_path}' not found. Please ensure the Dlib landmark predictor file is in the script directory.")
    exit()

# Load pre-trained eye state classification model
try:
    eye_model = load_model(eye_model_path)  # Pre-trained CNN model for eye state
except Exception as e:
    print(f"Error loading eye state model: {e}")
    exit()

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(landmark_model_path)
except Exception as e:
    print(f"Error loading Dlib landmark predictor: {e}")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Function to preprocess eye region for AI model
def preprocess_eye(eye_region):
    try:
        eye = cv2.resize(eye_region, (24, 24))  # Resize to match model input
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        eye = eye / 255.0  # Normalize pixel values
        eye = img_to_array(eye)  # Convert to array
        eye = np.expand_dims(eye, axis=0)  # Expand dimensions for model
        return eye
    except Exception as e:
        print(f"Error in eye preprocessing: {e}")
        return None

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

        # Extract eye regions
        left_eye_region = frame[landmarks[37][1]:landmarks[41][1], landmarks[36][0]:landmarks[39][0]]
        right_eye_region = frame[landmarks[43][1]:landmarks[47][1], landmarks[42][0]:landmarks[45][0]]

        # Preprocess eye regions and predict state
        left_eye_preprocessed = preprocess_eye(left_eye_region)
        right_eye_preprocessed = preprocess_eye(right_eye_region)

        if left_eye_preprocessed is not None and right_eye_preprocessed is not None:
            try:
                left_eye_state = np.argmax(eye_model.predict(left_eye_preprocessed, verbose=0)[0])
                right_eye_state = np.argmax(eye_model.predict(right_eye_preprocessed, verbose=0)[0])
            except Exception as e:
                print(f"Error in eye prediction: {e}")
                continue

            # Update states based on predictions
            if left_eye_state == 0 or right_eye_state == 0:  # Closed eyes
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    winsound.Beep(2500, 1000)  # Alert sound for sleeping
            elif left_eye_state == 1 or right_eye_state == 1:  # Partially closed eyes
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)
                    winsound.Beep(2000, 500)  # Alert sound for drowsy
            else:  # Open eyes
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

        # Display status
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Display video frames
    cv2.putText(frame, "Press ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("AI Drowsiness Detector", frame)

    # Exit on pressing ESC
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
