import cv2
import dlib
from scipy.spatial import distance as dist

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def alert_driver(state):
    if state == "drowsy":
        print("Drowsiness Detected! Please take a break.")
        # Optional: playsound('alert_sound.mp3') for an audio alert

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

        landmarks = predictor(gray, face)
        
        left_eye = landmarks.part(36:42)
        right_eye = landmarks.part(42:48)

        leftEAR = calculate_ear(left_eye)
        rightEAR = calculate_ear(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    alert_driver("drowsy")
        else:
            COUNTER = 0
            ALARM_ON = False

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
