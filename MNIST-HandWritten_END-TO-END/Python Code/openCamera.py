import time

import cv2
import numpy as np
import urllib.request
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


def main():
    # Replace the URL with your own IPwebcam shot.jpg IP:port
    url = 'http://192.168.1.70:8080/shot.jpg'

    loaded_model = load_model('final_model.h5')

    while True:
        # Use urllib to get the image from the IP camera
        imgResp = urllib.request.urlopen(url)

        # Numpy to convert into a array
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

        # Finally decode the array to OpenCV usable format ;)
        img = cv2.imdecode(imgNp, -1)
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ""

        if (len(contours)) > 0:
            contours = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contours) > 2500:
                x, y, w, h = cv2.boundingRect(contours)
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(1, 28, 28, 1)
                ans1 = loaded_model.predict(newImage)
                ans1 = ans1.tolist()
                ans1 = ans1[0].index(max(ans1[0]))

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(img, " Deep Network : " + str(ans1), (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # put the image on screen
        cv2.imshow('IPWebcam', img)
        cv2.imshow("Contours", thresh)

        # To give the processor some less stress
        #time.sleep(0.1)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
