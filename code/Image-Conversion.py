import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import numpy as np
import cv2
from PIL import Image

import os

from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import html
import torch
import torchvision
import torchvision.transforms as transforms
from drawToRobot import ImageToTxt

from helper_demo import helper


def load_model(model_name):
    opt = helper.setup_options()
    opt.name = model_name
    model = create_model(opt)
    model.isTrain = False
    model.setup(opt)

    return model


def get_webcam(model, mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    return img


def crop_face_from_webcam(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    for (x, y, w, h) in face:
        padding = int(0.2 * w)
        face = frame[y-padding:y+h+padding, x-padding:x+w+padding]
    return face


def face_to_goblin(frame, model):
    cvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    pil_img = Image.fromarray(cvframe)
    pil_img = pil_img.resize([256, 256])
    img = transform(pil_img)
    img = img.view(1, 3, 256, 256)
    img_A = helper.to_image(img)
    img_B = model.gen_B(img, 1)
    img_B = helper.to_image(img_B)
    img_AB = helper.concatenate([img_A, img_B])

    plt.axis('off')
    plt.title('Generated goblin face')
    plt.imshow(img_AB)
    plt.show()
    return img_B


def compare_faces(normal, goblin, lines, edges):
    # Create comparison plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    ax = axes.ravel()

    ax[0].imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB), cmap='gray')
    ax[0].set_title('Original image', fontsize=10)

    ax[1].imshow(goblin, cmap='gray')
    ax[1].set_title('CycleGAN to goblin face', fontsize=10)

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim(0, normal.shape[1])
    ax[2].set_ylim(normal.shape[0], 0)
    ax[2].set_title('Probabilistic Hough Lines', fontsize=10)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

def main():
    model = load_model("goblin")
    img = get_webcam(model)
    face = crop_face_from_webcam(img)
    goblin = face_to_goblin(face, model)
    lines, edges = ImageToTxt.image_to_txt(goblin)
    compare_faces(face, goblin, lines, edges)

if __name__ == '__main__':
    main()