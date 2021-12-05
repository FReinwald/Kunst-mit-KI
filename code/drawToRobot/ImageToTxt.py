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


def image_to_txt(img):
    # Constants
    # path_to_file_folder = './images/'
    # txt_file_name = 'testimg01.png'
    # path_to_file = path_to_file_folder+txt_file_name

    # Read image
    # img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = feature.canny(img, sigma=2.75)
    # based on: https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html#sphx-glr-auto-examples-edges-plot-line-hough-transform-py
    lines = probabilistic_hough_line(edges, line_length=2, line_gap=3)
    data_2_d = np.array(lines)

    # create oneline drawing
    combine_lines(data_2_d)

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


def combine_lines(data_2_d):
    start_point = np.array([0, 0])
    data_out = np.array([[100, 100, drawing_lift], [100, 100, drawing_lift]])
    # reshape array to be 2-dim
    data_2_d = np.reshape(data_2_d, (-1, 2))
    # add z-Data
    data_2_d = np.insert(data_2_d, 2, drawing_plane_z_level, axis=1)

    rows, columns = data_2_d.shape
    # iterate trough array to form new online
    for i in range(0, rows, 2):
        idx, dist = find_closest_point(start_point, data_2_d)
        # find second point belonging to hough line
        if idx % 2 == 0:
            idx2 = idx+1
        else:
            idx2 = idx-1
        # Lift pen if distance is too long
        if dist > 60:
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

    # idea - remove last 20% to prevent crossing over whole image
#    rows, columns = data_out.shape
#    data_out = np.delete(data_out, range(int(1), int(rows*0.4)), 0)
#    rows, columns = data_out.shape
#    data_out = np.delete(data_out, range(int(rows-1*0.4), int(rows-1)), 0)

    # Rescale x and y
    data_out[:, 0] = data_out[:, 0] * (x_max_drawing_value - x_zero_offset) / max(
        data_out[:, 0]) + x_zero_offset
    data_out[:, 1] = data_out[:, 1] * (y_max_drawing_value - y_zero_offset) / max(
        data_out[:, 1]) + y_zero_offset

    data_out[0, 2] = drawing_lift
    data_out = np.vstack((data_out, endpoint))
    data_out = data_out.flatten()
    f = open("robotmove_oneline.txt", "w")
    np.savetxt(f, data_out, "%d")
    f.close()


def find_closest_point(start_point, data_2_d):
    x = data_2_d[:, 0]
    y = data_2_d[:, 1]
    # calculate distance to startpoint
    distance = np.sqrt(np.square(x-start_point[0])+np.square(y-start_point[1]))
    # find index for closest point
    idx = np.argmin(distance)
    min_distance = distance[idx]
    return idx, min_distance
