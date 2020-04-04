import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     face_values = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
#     for x, y, w, h in face_values:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
#         roi=gray[y:y+h, x:x+w]
#         roi_c=frame[y:y+h, x:x+w]
#         eye_values=eye_cascade.detectMultiScale(roi)
#         for i,j,k,l in eye_values:
#             cv2.rectangle(roi_c, (i, j), (i + k, j + l), (0, 0, 0), 2)
#             #cv2.circle(frame,(i,j),2,(255,255,255),3)
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
img=cv2.imread("virat4.jpg")
cv2.imshow("",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_values = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
for x, y, w, h in face_values:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    roi = gray[y:y + h, x:x + w]
    roi_c = img[y:y + h, x:x + w]
    eye_values = eye_cascade.detectMultiScale(roi)
    for i, j, k, l in eye_values:
        cv2.rectangle(roi_c, (i, j), (i + k, j + l), (0, 0, 0), 2)
        # cv2.circle(frame,(i,j),2,(255,255,255),3)
cv2.imshow("result",img)
cv2.waitKey()
