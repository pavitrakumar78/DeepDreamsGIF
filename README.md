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
```python deep-dream.py [-inp INPUT] [-oct OCTAVES] [-oct_s OCTAVE_SCALE]
                     [-itr ITERATIONS] [-j JITTER] [-ss STEP_SIZE] [-l LAYER]
                     [-gpu GPU] [-dpr GO_DEEPER] [-scale_co SCALE_COEFFICIENT]
```
The arguments are pretty much self-explanatory.
This script can convert a GIF to a deep-dream version of itself by de-constructing the GIF into frames, applying the deepdream function on each of the frames and put it back into a GIF.  The convertion of GIF to frames and vice-versa is done by [gifsicle](https://www.lcdf.org/gifsicle/) - an open-source command-line tool for creating, editing, and getting information about GIF images and animations. 
This script can also convert a single image into a gif by repeatedly scaling (zooming) and applying the deep dream function on the previous frames to create an "inception" like gif which keeps going deeper and deeper. This mode is default if the input is an image, the depth of scaling is controlled by -dpr functions, by default it zooms in a scale of -scale_co for 50 frames.

###





