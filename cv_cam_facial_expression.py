from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import time

print("*******************************************")
print("Starting the python file!!")
print("*******************************************")

#emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion =  ['Confused', 'Happy', 'Stressed', 'Tran']

font = cv2.FONT_HERSHEY_SIMPLEX

def cam_run():
    print("-----------------")
    print("STARTING CAM")
    print("-----------------")
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    model = keras.models.load_model("new_model_f1.h5")   
    emotions = ["Tran"]
    with tf.device('/gpu:0'):
        start = time.time()
        while(True):
            ret, img = cap.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    face_component = gray[y:y+h, x:x+w]
                    fc = cv2.resize(face_component, (48, 48))
                    inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
                    inp = inp/255.
                    prediction = model.predict_proba(inp)
                    em = emotion[np.argmax(prediction)]
                    cv2.putText(img, em, (x, y), font, 1, (0, 255, 0), 2)
                    emotions.append(em)
                cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        end = time.time()
        
    dt = end - start
    scale = dt/len(emotions)

    stressed = emotions.count("Stressed")
    happy = emotions.count("Happy")
    confused = emotions.count("Confused")

    em = "temp"
    count = 0
    initial = 0
    Hemotion = {"Stressed":[], "Happy":[], "Confused":[], "Tran": []}

    for i in range(len(emotions)):
        if em == emotions[i]:
            count += 1
        else:
            if count >= 15:
                Hemotion[em].append((initial*scale, count*scale))
            count = 0
            initial=i
            em = emotions[i]
            
    print("----------------------------------------------------")

    print(Hemotion)
            
    print("----------------------------------------------------")
            
    print("STATS")
    print("stressed: " + str(100*stressed/len(emotions)))
    print("happy: " + str(100*happy/len(emotions)))
    print("confused: " + str(100*confused/len(emotions)))

    print("----------------------------------------------------")

    cap.release()
    cv2.destroyAllWindows()


    
if __name__=="__main__":
    cam_run()