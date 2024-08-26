import cv2
import mediapipe as mp
import math
import pickle
import numpy as np
import warnings
import pyttsx3
import time
from collections import Counter
from spellchecker import SpellChecker

warnings.filterwarnings('ignore')

Sign = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space']

# Function to calculate distance
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# Load the saved KNN model
with open('sign_gesture.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 125)

# Initialize spell checker
spell = SpellChecker()

# Start video capture
cap = cv2.VideoCapture(0)

# Variables for word formation
word = ''
last_gesture_time = time.time()
gesture_timeout = 2  # seconds

# Variables for prediction confirmation
frame_buffer = []
buffer_size = 4

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            # mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate distances between landmarks
            distances = []
            for i in range(len(hand_landmarks.landmark)):
                for j in range(i + 1, len(hand_landmarks.landmark)):
                    point1 = hand_landmarks.landmark[i]
                    point2 = hand_landmarks.landmark[j]
                    distance = calculate_distance(point1, point2)
                    distances.append(distance)
            
            # Reshape and scale distances to fit model input
            distances = np.array(distances).reshape(1, -1)
            
            # Predict gesture using the loaded model
            gesture = knn_model.predict(distances)
            
            # Add the prediction to the frame buffer
            frame_buffer.append(gesture[0])
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Confirm the letter if we have enough frames
            if len(frame_buffer) == buffer_size:
                most_common_gesture = Counter(frame_buffer).most_common(1)[0][0]
                sign = Sign[most_common_gesture]

                # Display the letter on the image
                cv2.putText(img, f'Letter: {sign}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 250), 2, cv2.LINE_AA)
              
    cv2.imshow("Hand Sign Language", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
