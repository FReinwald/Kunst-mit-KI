# This is a fork of [face-to-cartoon](https://github.com/fs2019-atml/face-to-cartoon).
The projekt was modified with a new model for generation of goblin faces. The generated images are also converted to be drawn by a robot.
Significant changes are either marked in the code with my name or added via new files.

## Added files
[anaconda-env.yaml](anaconda-env.yaml): An anaconda environment to be used on windows \
[linux-conda-env.yaml](linux-conda-env.yaml): An anaconda environment to be used on linux \
[goblin](code/checkpoints/goblin): A trained model to produce goblin faces from human faces \
[Image_preprocessing.py](code/datasets/image_preprocessing.py): A tool to preprocess data for training \
[ImageToTxt.py](code/drawToRobot/ImageToTxt.py): Transfers an image to a txt that represent hough lines and the robotpath \
[RobotSimulation.py](code/drawToRobot/RobotSimulation.py): Shows robot movement in 3D, calculated from a txt input \
[robotmove_oneline.txt](code/robotmove_oneline.txt): Txt output in a more modern format (x1 y1 x2 y2; ...) \
[robotmove_alt.txt](code/robotmove_alt.txt): Current txt output (x; y; z; x; y; ...) \
[Image-Conversion.py](code/Image-Conversion.py): Caputres webcam and converts image to goblin and then txt file \
## Modified files
[cycle_gan_model.py](code/models/cycle_gan_model.py): Added support for windows paths (default is linux, windows option commented out) \
[base_options.py](code/options/base_options.py): Changed training parameters \

## Running the program
run [Image-Conversion.py](/code/Image-Conversion.py)


## Code
More details to the code are in `code/README.md`.
