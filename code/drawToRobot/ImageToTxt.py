import numpy as np
import cv2
from skimage import feature
from skimage.transform import probabilistic_hough_line
drawing_plane_z_level = 108
drawing_lift = 135
x_zero_offset = 145
x_max_drawing_value = 235
y_zero_offset = 180
y_max_drawing_value = 270
endpoint = [215, 200, 118]
max_drawing_value = 145
# control how often the robot lifts the arm while drawing
max_distance = 1000


def image_to_txt(img):
    # Constants for debugging
    # path_to_file_folder = './images/'
    # txt_file_name = 'testimg01.png'
    # path_to_file = path_to_file_folder+txt_file_name
    # Read image
    # img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # canny edge detection
    edges = feature.canny(img, sigma=2.75)
    # hough lines from canny
    lines = probabilistic_hough_line(edges, line_length=4, line_gap=3)
    data_2_d = np.array(lines)

    # create oneline drawing
    data_out, data_out_flat = combine_lines(data_2_d, img.shape)

    return lines, edges, data_out, data_out_flat


def combine_lines(data_2_d, imgshape):
    start_point = np.array([0, 0])
    data_out = np.array([[100, 100, drawing_lift], [100, 100, drawing_lift]])
    # reshape array to be 2-dim
    data_2_d = np.reshape(data_2_d, (-1, 2))
    # add z-Data
    data_2_d = np.insert(data_2_d, 2, drawing_plane_z_level, axis=1)

    rows, columns = data_2_d.shape
    # iterate trough array to form new oneline
    for i in range(0, rows, 2):
        idx, dist = find_closest_point(start_point, data_2_d)
        # find second point belonging to hough line
        if idx % 2 == 0:
            idx2 = idx+1
        else:
            idx2 = idx-1
        # Lift pen if distance is too long
        if dist > max_distance:
            lift1 = np.array([data_out[-1, 0], data_out[-1, 1], drawing_lift])
            data_out = np.vstack((data_out, lift1))
            lift2 = np.array([data_2_d[idx, 0], data_2_d[idx, 1], drawing_lift])
            data_out = np.vstack((data_out, lift2))
        # add this line to output data
        data_out = np.vstack((data_out, data_2_d[idx, :]))
        data_out = np.vstack((data_out, data_2_d[idx2, :]))
        # use last point/position as new startPoint for next loop and find closest point to it again
        start_point = data_2_d[idx2, :]
        # delete output lines from data_2_d so they won't be drawn twice
        data_2_d = np.delete(data_2_d, [idx, idx2], 0)

    # Rescale x and y
    data_out[:, 0] = data_out[:, 0] * (x_max_drawing_value - x_zero_offset) / imgshape[0] + x_zero_offset
    data_out[:, 1] = data_out[:, 1] * (y_max_drawing_value - y_zero_offset) / imgshape[1] + y_zero_offset

    data_out[0, 2] = drawing_lift
    # add endpoint
    data_out = np.vstack((data_out, endpoint))
    data_out_flat = data_out.flatten()
    # Write to txt
    f = open("robotmove_oneline.txt", "w")
    np.savetxt(f, data_out_flat, "%d")
    f.close()
    return data_out, data_out_flat


def find_closest_point(start_point, data_2_d):
    x = data_2_d[:, 0]
    y = data_2_d[:, 1]
    # calculate distance to startpoint
    distance = np.sqrt(np.square(x-start_point[0])+np.square(y-start_point[1]))
    # find index for closest point
    idx = np.argmin(distance)
    min_distance = distance[idx]
    return idx, min_distance
