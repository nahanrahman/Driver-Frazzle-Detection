import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
src_img = cv2.imread('file_name.jpg')

# Convert into grayscale
gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(src_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Output', src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
