import cv2
import glob
import os

inputFolder = 'goblin/unnamed'
os.mkdir('preprocessed')
path = 'preprocessed'

dim = (280, 280)

i = 1
ii = 0
for img in glob.glob(inputFolder + "/*.jpg"):
    image = cv2.imread(img)
    # cv2.imshow('image', image)
    # cv2.waitKey(30)
    imgResized = cv2.resize(image, dim)
    cv2.imwrite(os.path.join(path, "%02i_%04i.jpg") % (ii, i), imgResized)
    i += 1
    # Nine blocks a 70 pictures
    if i == 71:
        ii += 1
        i = 1

# Path for image training: ./datasets/faces/cartoon_conditional
