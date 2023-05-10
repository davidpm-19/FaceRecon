# Importar módulos
import cv2

# Cargar modelo entrenado con los datos
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Model/Training_data.yml')

# Inicializar variables de uso general
x, y, w, h = 0, 0, 0, 0
tempLabel = ''


# Función de detección y dibujo del marco
def face_detector(img):
    global x, y, w, h, roi
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting colored image to grayscale

    # Cargar el clasificador prefabricado
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Obtener coordenadas de las caras
    faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=7)

    if faces == ():
        return img, []

    # Dibujo del marco alrededor de las caras
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (500, 500))

    return img, roi


# Diccionario para mostrar de forma más entendible los nombres de caras reconocidas
name = {0: 'Dave', 1: 'Marti', 2: 'Dani'}

# Captura de video
cap = cv2.VideoCapture(1)

while True:
    ret, img_frame = cap.read()  # Leer fotograma de camara

    image, req_face = face_detector(img_frame)  # llamada de función

    try:
        req_face = cv2.cvtColor(req_face, cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(req_face)  # Predecir el nombre
        print('Confidence :', confidence)
        print('Label :', label)

        tempLabel = label

        face_label = name[label]  # Nombre para colocar en frame

        #
        if (label == tempLabel) and (confidence < 50):
            cv2.putText(image, face_label, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0),2)  # mostrado de nombre
        else:
            cv2.putText(image, 'Unknown', (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255),2)  # mostrado de unknown cuando la cara no es reconocida pero se detecta cara
        cv2.imshow('Face Recognizer', image)

    except:
        cv2.putText(image, '', (50, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255),2)
        cv2.imshow('Face Recognizer', image)

    if cv2.waitKey(1) == 13:  # solo salir si se pulsa la tecla enter
        break

cap.release()  # libera la camara
cv2.destroyAllWindows()
