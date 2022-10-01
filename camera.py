import cv2

facial_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:

  ok, frame = video_capture.read()
  image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
  detection = facial_detector.detectMultiScale(image_gray, minSize=(150,150))	
 
  for x, y, w, h in detection:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255))
    cv2.imshow('Video', frame)
		
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break 

video_capture.release()
cv2.destroyAllWindows()
