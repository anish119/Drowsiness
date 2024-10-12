import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import webbrowser
import warnings
import time
warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 30
COUNTER = 0
VIDEO_OPENED = False    
DROWSINESS_DETECTED = False


YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley'  

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 374]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

cv2.namedWindow('Drowsiness Detector', cv2.WND_PROP_FULLSCREEN)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Process each face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Get coordinates for the left and right eyes
                left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE])
                right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE])

                # Calculate the EAR for both eyes
                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)

                # Average the EAR of both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # Check if EAR is below the threshold
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    # If eyes closed for enough consecutive frames, trigger the alert and open video
                    if COUNTER >= EYE_AR_CONSEC_FRAMES and not VIDEO_OPENED : 
                            webbrowser.open(YOUTUBE_VIDEO_URL)
                            VIDEO_OPENED = True
                            DROWSINESS_DETECTED = True
                            cv2.putText(frame, "YOU ARE SLEEPING DUDE,", (5, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "LISTEN TO THIS MUSIC AND WAKE THE HELL UP!!!!!", (5, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    COUNTER = 0
                    if DROWSINESS_DETECTED:
                        VIDEO_OPENED = False
                        DROWSINESS_DETECTED = False

                # Draw the eye landmarks on the frame
                for i in LEFT_EYE + RIGHT_EYE:
                    x = int(landmarks[i].x * frame.shape[1])
                    y = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        
        cv2.imshow('Drowsiness Detector', frame)
        
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Application interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows() 
