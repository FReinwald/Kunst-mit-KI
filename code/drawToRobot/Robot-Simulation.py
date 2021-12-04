#Robot-Simulation
#This file converts a txt file of coordinates to a visual representation of the robot movement

import numpy as np
import matplotlib.pyplot as plt

#Robot specific drawing area
xmax = 235
ymax = 270
zmax = 150
drawing_plane = 108

#set path to file
path_to_file_folder = 'C:/Users/Fabian/OneDrive - stud.hs-heilbronn.de/HHN/WS2122/Semesterarbeit/face-to-cartoon/code/'
txt_file_name = 'robotmove_oneline.txt'
path_to_file = path_to_file_folder+txt_file_name

#read txt file with path coordinates
coords = np.loadtxt(path_to_file)

#slice txt in x,y and z
x = coords[0::3]
y = coords[1::3]
z = coords[2::3]

#direction of vectors
x_dir = x[1:]
x_dir = np.append(x_dir, 0)-x
x_dir[-1] = 0
y_dir = y[1:]
y_dir = np.append(y_dir, 0)-y
y_dir[-1] = 0
z_dir = z[1:]
z_dir = np.append(z_dir, 0)-z
z_dir[-1] = 0

#todo: find way to properly color quivers depending if the robot draws or not
bool_not_drawing = z > drawing_plane
c = [0, 0, 0]

#drawing of plot
ax = plt.figure().add_subplot(projection='3d')
ax.set(xlim=(145, xmax), ylim=(180, ymax), zlim=(100, zmax))
ax.quiver(x, y, z, x_dir, y_dir, z_dir, colors=c, normalize=False, arrow_length_ratio=0.05)
# Disable
# ax.quiver(x, y, z, x_dir, y_dir, z_dir, colors=c, normalize=False, arrow_length_ratio=0.05)
plt.show()
