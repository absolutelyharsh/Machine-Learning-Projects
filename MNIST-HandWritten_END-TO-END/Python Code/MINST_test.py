import cv2
from keras.models import load_model


# Function to test the image
def run_live_example():
    #comment
    # read the image. webcam module can also be added
    image = cv2.imread('2.jpg')

    # convert image into grayscale for thresholding
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold the image with inverse flag for binarization
    ret, thresh_image = cv2.threshold(image, 95, 255, cv2.THRESH_BINARY_INV)

    # display the image
    cv2.imshow('thresh_image', thresh_image)
    cv2.waitKey(0)

    # resithe image as per the model requirement
    resized_thresh_image = cv2.resize(thresh_image, (28, 28))

    resized_thresh_image = resized_thresh_image.reshape(1, 28, 28, 1)

    resized_thresh_image = resized_thresh_image.astype('float32')

    resized_thresh_image = resized_thresh_image / 255.0

    # load the model
    model = load_model('final_model.h5')

    # prediction
    digit = model.predict_classes(resized_thresh_image)

    print(digit[0])


run_live_example()

cv2.destroyAllWindows()
