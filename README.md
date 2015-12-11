# DeepDreamsGIF

Python script to convert any GIF to a REALLY trippy GIF using Google DeepNet model.


Requirements:

- Python 2.7
- caffe and its dependencies


This is based on Google's [Blog](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html) post about neural networks. The official repo with the code for generating images is available [here](https://github.com/google/deepdream).

I have used the same code as the one found in Google's repo with slight modifications to make it work for both GIFs and images.  All the parameters can be tweaked right in the command line arguments.

##File Descriptions:

###deep-dream.py

Usage:  
```
python deep-dream.py -inp INPUT [-oct OCTAVES] [-oct_s OCTAVE_SCALE]
                     [-itr ITERATIONS] [-j JITTER] [-ss STEP_SIZE] [-l LAYER]
                     [-gpu GPU] [-dpr GO_DEEPER] [-scale_co SCALE_COEFFICIENT]
```
The arguments are pretty much self-explanatory.
This script can convert a GIF to a deep-dream version of itself by de-constructing the GIF into frames, applying the deepdream function on each of the frames and put it back into a GIF.  The convertion of GIF to frames and vice-versa is done by [gifsicle](https://www.lcdf.org/gifsicle/) - an open-source command-line tool for creating, editing, and getting information about GIF images and animations. 
This script can also convert a single image into a gif by repeatedly scaling (zooming) and applying the deep dream function on the previous frames to create an "inception" like gif which keeps going deeper and deeper. This mode is default if the input is an image, the depth of scaling is controlled by -dpr functions, by default it zooms in a scale of -scale_co for 50 frames.

This script only uses the (GoogleLeNet)[https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet] model, so you might need to modify the modify the script to change location of those files.

###try-layers.py
```
python try-layers.py -inp INPUT -deploy_file DEPLOY_FILE -caffe_model
                     CAFFE_MODEL [-p PRINT_ONLY] [-oct OCTAVES]
                     [-oct_s OCTAVE_SCALE] [-itr ITERATIONS] [-j JITTER]
                     [-s STEP_SIZE] [-gpu GPU] [-scale_coef SCALE_COEFFICIENT]
```
Almost same as the previous script, but this script allows you to use your own custom models for generating images and also you get to know about the layers present in the the model for use with the -l (layer) param in previous script.  Only images are accepted as input. If -p param is set to 1 (default), the only the names of the layer will be printed in the console, if -p is set to 0 then the image output for each of the layer is saved in a 'frames' folder in the same directory. Regardless of the value of -p, a text file with the names of all the layers is always saved in the current directory on executing the script.

Usefull links to get various models:

- [GoogleLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) (Used in previous script)
- [MIT's Places](http://places.csail.mit.edu/downloadCNN.html)
- [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) (A list of many other models trained on various objects)


###gifsicle.exe and gifsicle.dif
These files are from their official [page](https://www.lcdf.org/gifsicle/) the main script depends on these files to construct and de-construct GIFs. Place these files in the same folder as the script you are executing.

##Examples:
```
python deep-dream.py -input bear.gif -gpu 1
```
Original:
![bear.gif](https://raw.github.com/pavitrakumar78/DeepDreamsGIF/examples/bear.gif)
After applying deepdream:
![bear-dream.gif](https://raw.github.com/pavitrakumar78/DeepDreamsGIF/examples/bear-dream.gif)


A much scarier example..
![train.gif](https://raw.github.com/pavitrakumar78/DeepDreamsGIF/examples/train.gif)
After applying deepdream:
![train-dream.gif](https://raw.github.com/pavitrakumar78/DeepDreamsGIF/examples/train-dream.gif)

```
python deep-dream.py -input flowers.jpg -gpu 1 -dpr 100
```
![flower-dream-deepr.gif](https://raw.github.com/pavitrakumar/DeepDreamsGIF/examples/flowers-dream-deepr.gif)

