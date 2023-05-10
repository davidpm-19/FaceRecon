import os
import cv2
import numpy as np
# Path where image samples are stored
data_path = 'Face_samples_dataset'

# Face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Lists for storing our training data and labels associated
training_data, labels = [], []

for path, subdirname, filenames in os.walk(data_path):
    print('path:',path)
    print('subdirname:',subdirname)
    print('filenames:',filenames)

    for file_name in filenames:
        if file_name.startswith('.'):
            print('Skipping system file')
            continue

        f_id = os.path.basename(path)
        img_path = os.path.join(path,file_name)
        print('img_path:',img_path)
        print('f_id:',f_id)

        test_img = cv2.imread(img_path)
        if test_img is None:
            print('Image not loaded properly !!!')
            continue

        test_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(test_gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) != 1:
            continue  # Since we are assuming only single person images are being fed to classifier

        for (x, y, w, h) in faces:
            roi_gray = test_gray[y:y + h, x:x + w]  # cropping region of interest i.e. face area

        roi_gray = cv2.resize(roi_gray, (500, 500))  # Resizing the cropped region

        # Preparing data for training
        training_data.append(roi_gray)  # Appending the face portions/region of interest and creating the training data
        labels.append(int(f_id))  # Appending the labels associated with the faces detected

# creating model using LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Training the data
model.train(np.asarray(training_data), np.asarray(labels))

# Saving the trained model
model.save('Model/Training_data.yml')

print('Model training complete !!!')

