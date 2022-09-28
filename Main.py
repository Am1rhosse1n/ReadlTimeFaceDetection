#CascadeClassifier "Rapid Object Detection using a Boosted Cascade of Simple Features"

import cv2


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

if (cap.isOpened()):
        print('Everything is ok')
        while(True):
        
            ret, frame = cap.read()    
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray',gray)
            #gray = cv2.equalizeHist(gray)
            #cv2.imshow('hist',gray)
            
            
            #OpenCV Pre-Trained classifiers
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            
            faces = face_cascade.detectMultiScale(gray,1.25,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_RGB = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray,1.25,8)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_RGB,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
            
            try:
                FaceNum = faces.shape[0]
            except:
                FaceNum = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            if FaceNum <= 1:
                frame = cv2.putText(frame,str(FaceNum)+' Face is being Detected',(20,450), font, 0.8,(255,255,255),2,cv2.LINE_AA)
            else:
                frame = cv2.putText(frame,str(FaceNum)+' Faces are being Detected',(20,450), font, 0.8,(255,255,255),2,cv2.LINE_AA)
          
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
               break
else:
    print('There is a problem')

cap.release()
cv2.destroyAllWindows()

