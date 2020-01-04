import cv2
from autocropper import AutoCropper
import os
import numpy as np

def Ocisti_sliku(img):  
    ac = AutoCropper(img)
    ac.max_border_size = 36 	# defaults to 300
    ac.safety_margin = 0		# defaults to 4, removes extra pixels from the sides to make sure no black remains
    ac.tolerance = 0			# defaults to 4, a gray value is more likely to be considered black when you increase the tolerance

    # go!
    result = ac.autocrop()

    a = np.asarray(result)
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print(img_binary)
    shape = img_binary.shape
    #print(shape[0])

    #print(img_binary[1][1])
    max_remove = shape[1]-1
    for i in range(0, shape[0]):
        j = 0
        while img_binary[i][j] == 0 and j < max_remove:
            img_binary[i][j] = 255
            j = j+1

    for i in range(shape[0]-1, 0, -1):
        j = shape[1] - 1
        while img_binary[i][j] == 0 and j > shape[1]-max_remove:
            img_binary[i][j] = 255
            j = j-1
        

    #im = cv2.imread('image.jpg')
    row, col = img_binary.shape[:2]
    bottom = img_binary[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 4
    border = cv2.copyMakeBorder(
        img_binary,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )

    kernel = np.ones((3,3),np.uint8)
    denoised = cv2.morphologyEx(border, cv2.MORPH_CLOSE, kernel)

    erode_kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(denoised,erode_kernel,iterations = 1)
    return erosion



#img = cv2.imread("images/image22-gs.jpg", cv2.IMREAD_COLOR)
#PATH_TO_RESULT_IMAGES_DIR = 'borderless'
#nova = Ocisti_sliku(img)
#print(nova)
#cv2.imwrite(os.path.join(PATH_TO_RESULT_IMAGES_DIR, 'image1.jpg'), nova)  