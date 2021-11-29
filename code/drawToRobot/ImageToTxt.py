import numpy as np
import cv2
from skimage import feature
from skimage.transform import probabilistic_hough_line


def image_to_txt(img):
    # Constants
    # path_to_file_folder = './images/'
    # txt_file_name = 'testimg01.png'
    # path_to_file = path_to_file_folder+txt_file_name
    drawing_plane_z_level = 108
    drawing_lift = 135
    x_zero_offset = 145
    x_max_drawing_value = 235
    y_zero_offset = 180
    y_max_drawing_value = 270
    endpoint = [215, 200, 118]
    max_drawing_value = 145

    # Read image
    # img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = feature.canny(img, sigma=2.75)
    # based on: https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html#sphx-glr-auto-examples-edges-plot-line-hough-transform-py
    lines = probabilistic_hough_line(edges, line_length=1, line_gap=1)
    data_2_d = np.array(lines)

    # iterate trough 2xN array if (2,N) = (1,N+1) don't lift pen. Otherwise add 2 points to lift pen
    data_2_d = np.insert(data_2_d, 2, drawing_plane_z_level, axis=2)
    temp = np.array([[0, 0], [215, 200]])
    move_data_complete = np.array([215, 200, drawing_lift])
    # Detect if lines start onto each other
    for x in data_2_d:
        if (x[1, 0] == temp[0, 0]) and (x[1, 1] == temp[0, 1]):
            move_data = x
        else:
            up = np.array([temp[1, 0], temp[1, 1], drawing_lift])
            strafe = np.array([x[0, 0], x[0, 1], drawing_lift])
            move_data = np.vstack((up, strafe, x))
        temp = x
        move_data_complete = np.vstack((move_data_complete, move_data))

    # Rescale x and y
    move_data_complete[:, 0] = move_data_complete[:, 0] * (x_max_drawing_value - x_zero_offset) / max(
        move_data_complete[:, 0]) + x_zero_offset
    move_data_complete[:, 1] = move_data_complete[:, 1] * (y_max_drawing_value - y_zero_offset) / max(
        move_data_complete[:, 1]) + y_zero_offset

    # Add Endpoint
    move_data_complete = np.vstack((move_data_complete, endpoint))

    # One Row for each value
    move_data_complete = move_data_complete.flatten()

    # Write txt
    f = open("robotmove.txt", "w")
    np.savetxt(f, move_data_complete, '%d')
    f.close()

    return lines, edges



