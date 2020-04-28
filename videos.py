
import numpy as np
import cv2
import math
import imutils      # allows video editing
from imutils.object_detection import non_max_suppression

cap = cv2.VideoCapture(0)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

find_faces = False
avg_color = True
i = 0



while(cap.isOpened()):
    i += 1

    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        # out.write(frame)

        # resize
        # frame = imutils.resize(frame, width = 400)

        # convert to graysclae and equalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)


        if find_faces:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


            # change the second param (scaleFactor)
            faces = face_cascade.detectMultiScale(gray, 1.5, 5)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                # I only have 2 eyes, keeping this until 3+ eyes in video
                eyes = eyes[:2]
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if avg_color:

            average = frame.mean(axis=0).mean(axis=0)
            # print(average)
            # print("hi")

            pixels = np.float32(frame.reshape(-1, 3))

            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS

            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)

            dominant = palette[np.argmax(counts)]

            x,y,w,h = 5,5,100,100

            print(dominant)

            t2 = tuple(map(lambda x: math.ceil(x), dominant))

            print(t2)

            # checked that it works using this, try placing oddly colored objects in front of your camera
            # frame = cv2.rectangle(frame,(x,y),(x+w,y+h), t2,10)

            # print(dominant)

        cv2.imshow('frame',frame)
        #debugging
        if i == 3:
            # print(frame.shape)
            # print(len(frame[0]))
            # print(frame)
            # break
            pass


        # cv2.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
# console.log()
