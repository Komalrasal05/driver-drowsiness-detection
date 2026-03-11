import os
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

# --------------------------
# Initialize Mixer
# --------------------------
mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Alarm Path
alarm_path = os.path.join(BASE_DIR, "alarm.wav")
if not os.path.exists(alarm_path):
    print("ERROR: alarm.wav not found!")
    exit()

sound = mixer.Sound(alarm_path)

# --------------------------
# Eye Aspect Ratio
# --------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --------------------------
# Mouth Aspect Ratio (Yawn)
# --------------------------
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# --------------------------
# Thresholds
# --------------------------
eye_thresh = 0.25
eye_frame_check = 20

yawn_thresh = 0.6
yawn_frame_check = 15

eye_flag = 0
yawn_flag = 0

# --------------------------
# Load Face Detector
# --------------------------
detect = dlib.get_frontal_face_detector()

predictor_path = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    print("ERROR: shape_predictor file not found!")
    exit()

predict = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# --------------------------
# Start Webcam
# --------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found")
    exit()

# --------------------------
# Main Loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # ---------------- Eye Detection ----------------
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < eye_thresh:
            eye_flag += 1
            if eye_flag >= eye_frame_check:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.get_busy():
                    sound.play()
        else:
            eye_flag = 0

        # ---------------- Yawn Detection ----------------
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        if mar > yawn_thresh:
            yawn_flag += 1
            if yawn_flag >= yawn_frame_check:
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if not mixer.get_busy():
                    sound.play()
        else:
            yawn_flag = 0

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------
# Cleanup
# --------------------------
cap.release()
cv2.destroyAllWindows()
mixer.quit()