
import cv2

#importing cascades
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


#video object
video = cv2.VideoCapture(0)


while True:
    success, img = video.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(imgGray,1.3,5)

    #creating rectangle
    for (x,y,w,h) in face:
        img = cv2.rectangle(img,(x,y), (x + w, y + h), (0,0,255,),3)

    #capturnig each frame
    ret, frame = video.read()
    face = faceCascade.detectMultiScale
    

    #displays the current frame
    cv2.imshow("frame",img)

    

    #q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()