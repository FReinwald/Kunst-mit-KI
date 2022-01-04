# This is a fork of [face-to-cartoon](https://github.com/fs2019-atml/face-to-cartoon).
The projekt was modified with a new model for generation of goblin faces. The generated images are also converted to be drawn by a robot.
Significant changes are either marked in the code with my name or added via new files.

## Running the program
run [Image-Conversion.py](/code/Image-Conversion.py)

For changes to the original CycleGAN see below:

# Code notes
Our contributions are declared using the following comment:
`#### GROUP5 code ####` and `#### END of code ####`. Finding them using grep: `grep -rnw . -e 'GROUP5'`.

Most are found under
* `data/face_dataset.py` (our own data handling and augmentation incorporating landmarks)
* `model/cycle_gan_model.py` (adaptation to handle own data types and conditional networks)
* `model/networks.py` (the conditional architecture, the landmark network (LDNet), Own loss functions)
* `landmarks/*` (matlab script to label faces with landmarks)
* `cam.py` (demo script to use our model of the commandline)

## Acknowledgments
Our code is a fork of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
