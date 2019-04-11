# Pytorch-UNet
// TO DO

## Usage

### Prediction

You can easily test the output masks on your images via the CLI.

To see all options:
`python predict.py -h`

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

You can use the cpu-only version with `--cpu`.

You can specify which model file to use with `--model MODEL.pth`.

### Training

`python train.py -h` should get you started. A proper CLI is yet to be added.
## Warning
In order to process the image, it is split into two squares (a left on and a right one), and each square is passed into the net. The two square masks are then merged again to produce the final image. As a consequence, the height of the image must be strictly superior than half the width. Make sure the width is even too.

## Dependencies
This package depends on [pydensecrf](https://github.com/lucasb-eyer/pydensecrf), available via `pip install`.

## Notes on memory

The model has be trained from scratch on a GTX970M 3GB.
Predicting images of 1918*1280 takes 1.5GB of memory.
Training takes approximately 3GB, so if you are a few MB shy of memory, consider turning off all graphical displays.
This assumes you use bilinear up-sampling, and not transposed convolution in the model.
