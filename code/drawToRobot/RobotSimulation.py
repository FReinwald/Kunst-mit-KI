# Robot-Simulation
# This file converts a txt file of coordinates to a visual representation of the robot movement

import numpy as np
import matplotlib.pyplot as plt

# Robot specific drawing area
xmax = 235
ymax = 270
zmax = 150
x_zero_offset = 145
y_zero_offset = 180
drawing_plane = 108

# set path to file
path_to_file_folder = 'C:/Users/Fabian/OneDrive - stud.hs-heilbronn.de/HHN/WS2122/Semesterarbeit/face-to-cartoon/code/'
txt_file_name = 'robotmove_oneline.txt'
path_to_file = path_to_file_folder+txt_file_name

# read txt file with path coordinates
coords = np.loadtxt(path_to_file)

# slice txt in x,y and z
x3d = coords[0::3]
y3d = coords[1::3]
z3d = coords[2::3]


# direction of vectors
def calculate_direction(points):
    vdir = points[1:]
    vdir = np.append(vdir, 0)-points
    vdir[-1] = 0
    return vdir


x_dir = calculate_direction(x3d)
y_dir = calculate_direction(y3d)
z_dir = calculate_direction(z3d)

c = [0, 0, 0]

#drawing of 3d plot
ax3d = plt.figure().add_subplot(projection='3d')
ax3d.set(xlim=(x_zero_offset, xmax), ylim=(y_zero_offset, ymax), zlim=(100, zmax))
ax3d.quiver(x3d, y3d, z3d, x_dir, y_dir, z_dir, colors=c, normalize=False, arrow_length_ratio=0.05)

