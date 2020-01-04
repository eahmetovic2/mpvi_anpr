from PIL import Image , ImageDraw
import numpy as np
import os
import glob
import cv2
from CNN_model import CNN_model, daj_model
from segment_characters import segment_characters
from border_remove import Ocisti_sliku


def Character_recognition_multiple(path = 'images/*.jpg'):
    #model = CNN_model()
    model = daj_model()

    abeceda = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'J', 'K', 'M', 'O', 'T', '' ]

    images = []
    image_paths = glob.glob( path )
    for imagefile in image_paths:
        img = cv2.imread(imagefile, cv2.IMREAD_COLOR) 
        #print(imagefile[7:])
        ociscena = Ocisti_sliku(img)
        #PATH_TO_BORDERLESS_IMAGES_DIR = 'borderless'
        #cv2.imwrite(os.path.join(PATH_TO_BORDERLESS_IMAGES_DIR, imagefile[7:]), ociscena)  
        images.append( ociscena )

    index = 0
    index2 = 0
    output = []
    for im in images:
        char = segment_characters(im)
        registracija = []
        for i, ch in enumerate(char):
            #PATH_TO_RESULT_IMAGES_DIR = 'result'
            #cv2.imwrite(os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image{}{}.jpg'.format(index + 1, index2 + 1 )), ch)  
            test = cv2.resize(ch, (28, 28))
            test = test.reshape(1,28,28,1)
            y = model.predict_classes(test)
            y_ = y[0]

            registracija.append(abeceda[y_])
            index2 = index2 + 1
        output.append(''.join(registracija))
        index = index + 1
    print(output)

def Character_recognition(img):
    #model = CNN_model()
    model = daj_model()

    abeceda = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'J', 'K', 'M', 'O', 'T', '' ]

    ociscena = Ocisti_sliku(img)

    PATH_TO_BORDERLESS_IMAGES_DIR = 'borderless'
    cv2.imwrite(os.path.join(PATH_TO_BORDERLESS_IMAGES_DIR, 'image1.jpg'), ociscena)  

    karakteri = segment_characters(ociscena)
    registracija = []
    for i, ch in enumerate(karakteri):
        
        PATH_TO_RESULT_IMAGES_DIR = 'segmentation'
        cv2.imwrite(os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image{}.jpg'.format(i)), ch)  
        test = cv2.resize(ch, (28, 28))
        test = test.reshape(1,28,28,1)
        y = model.predict_classes(test)
        y_ = y[0]

        registracija.append(abeceda[y_])

    output = ''.join(registracija)
    print(output)
    return output