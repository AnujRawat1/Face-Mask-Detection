import os
import cv2 
from keras.models import load_model 

# Load Model
model = load_model("face_mask_detection_model.h5")

def detect_face_mask(img):
  y_pred = model.predict(img.reshape(1, 224, 224, 3))
  return y_pred[0][0]

def draw_label(img, text, pos, bg_color):

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)

    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    coordinates = haar.detectMultiScale(img)
    return coordinates

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

  # Check if the frame is successfully read
    if not ret:
        print("Error: Could not read frame. Check camera connection.")
        break  # Exit the loop if frame is not read

  # Call the detection method
    img = cv2.resize(frame, (224, 224))
    y_pred = detect_face_mask(img)


    coordinates = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    for x,y,w,h in coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
    
    if y_pred <= 0.5:
        draw_label(frame, 'Mask', (30,30), (0,255,0))
    else:
        draw_label(frame, 'No Mask', (30,30), (0,0,255))
    
    # if y_pred > 0.5 : 
    #     print('Person is not wearing a mask . Probability : ', y_pred*100,' % \n')
    # else:
    #     print('Person is wearing a mask . Probability : ',  y_pred*100,' % \n')


    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# cap.release()
cv2.destroyAllWindows()