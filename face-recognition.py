import cv2
import cv2.data

image = cv2.imread('gallery/facephoto.jpg')

#load pre trained cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Detect faces on the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(55,55))
#Detetct eyes on the image
eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(55,55))

#Draw rectangles in the face
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2)

#Draw rectangles in the eye
for (x,y,w,h) in eyes:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
