# This file handles the whole process converting an image from a webcam to an AI picture and a drawing file to the robot
# Code by Fabian Reinwald
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button
import torchvision.transforms as transforms
import numpy as np
import pyperclip as pc

from models import create_model
from drawToRobot import ImageToTxt
from helper_demo import helper


def load_model(model_name):
    opt = helper.setup_options()
    opt.name = model_name
    model = create_model(opt)
    model.isTrain = False
    model.setup(opt)

    return model


def get_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        # flip image for mirroring
        img = cv2.flip(img, 1)
        cv2.imshow('Press space to take a picture', img)
        # take picture by pressing space or escape
        if cv2.waitKey(1) == 32:
            break
    cv2.destroyAllWindows()
    # img = build_ui(cam)
    return img


def crop_face_from_webcam(frame):
    # Find face with haarcascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    for (x, y, w, h) in face:
        # increase border around face
        padding = int(0.2 * w)
        face = frame[y-padding:y+h+padding, x-padding:x+w+padding]
    return face


def face_to_goblin(frame, model):
    # image format changes
    cvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    pil_img = Image.fromarray(cvframe)
    pil_img = pil_img.resize([256, 256])
    img = transform(pil_img)
    img = img.view(1, 3, 256, 256)
    # applying model to image
    img_a = model.gen_B(img, 1)
    img_a = helper.to_image(img_a)

    return img_a


def compare_faces(face, goblin, lines, edges, data_out, data_robot_format):
    # Copy data_out to clipboard to be entered in robot controlling software
    out_txt = '\n'.join([f"{line}" for line in data_robot_format])
    pc.copy(out_txt)
    # Create comparison plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 4))
    fig.suptitle('Image conversion process - data for robot controller was copied to clipboard')
    ax = axes.ravel()

    # Original image
    ax[0].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), cmap='gray')
    ax[0].set_title('Original image', fontsize=10)

    # AI image
    ax[1].imshow(goblin, cmap='gray')
    ax[1].set_title('CycleGAN to goblin face', fontsize=10)

    # Edge detection
    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim(0, 256)
    ax[2].set_ylim(256, 0)
    ax[2].set_title('Probabilistic Hough Lines', fontsize=10)

    # Given multiple points, calculate to direction of the vector connection them for quiver plotting
    def calculate_direction(points):
        vdir = points[1:]
        vdir = np.append(vdir, 0) - points
        vdir[-1] = 0
        return vdir

    # Parameters of robot
    x_zero_offset = 145
    xmax = 235
    y_zero_offset = 180
    ymax = 270

    x = data_out[:, 0]
    y = data_out[:, 1]
    xd = calculate_direction(x)
    yd = calculate_direction(y)

    # Oneline drawing
    ax[3].imshow(edges * 0, alpha=0)
    ax[3].quiver(x, y, xd, yd, angles='xy', scale_units='xy', scale=1)
    ax[3].set(xlim=(x_zero_offset, xmax), ylim=(ymax, y_zero_offset))
    ax[3].set_title('Oneline drawing from Robot', fontsize=10)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()


def main():
    model = load_model("goblin")
    img = get_webcam()
    face = crop_face_from_webcam(img)
    goblin = face_to_goblin(face, model)
    lines, edges, data_out, data_robot_format = ImageToTxt.image_to_txt(goblin)
    compare_faces(face, goblin, lines, edges, data_out, data_robot_format)


if __name__ == '__main__':
    main()
