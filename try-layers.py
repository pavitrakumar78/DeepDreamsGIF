# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:36:12 2015

@author: Pavitrakumar
"""

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from PIL.Image import fromarray as img_fromarray
from google.protobuf import text_format
import os
import argparse
import caffe


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data 


def make_step(net, step_size=1.5, end='inception_5b/pool_proj', 
              jitter=32, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    #if clip:
    bias = net.transformer.mean['data']
    src.data[:] = np.clip(src.data, -bias, 255-bias) 

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_5b/pool_proj', jitter = 32,step_size=1.5):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end,step_size=step_size,jitter=jitter)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


def main(img_name,deploy_file,caffe_model,print_only,octaves=4,octave_scale=1.4,iterations=10,jitter=32,step_size=1.5,gpu=0,scale_coefficient=0.05):
    
    
    if gpu:
        caffe.set_mode_gpu()
    else:
        print "You are using CPU mode, this might take some time."
    
    if not print_only:
        if os.system("mkdir frames"):
            print "temp. output folder already exists, deleting exisisting one.."
            os.system("rm -r frames")
            os.system("mkdir frames")
    

    
    net_fn   = deploy_file
    param_fn = caffe_model
    
    #model_path = 'E:/Software/WinPython-64bit-2.7.10.3/python-2.7.10.amd64/caffe/models/finetune_flickr_style/' # substitute your path here
    #net_fn   = model_path + 'deploy.prototxt'
    #param_fn = model_path + 'finetune_flickr_style.caffemodel'
    
    #model_path = 'E:/Software/WinPython-64bit-2.7.10.3/python-2.7.10.amd64/caffe/models/bvlc_googlenet/' # substitute your path here
    #net_fn   = model_path + 'deploy.prototxt'
    #param_fn = model_path + 'bvlc_googlenet.caffemodel'
    
    #model_path = 'E:/Software/WinPython-64bit-2.7.10.3/python-2.7.10.amd64/caffe/models/vgg_face_caffe/' # substitute your path here
    #net_fn   = model_path + 'VGG_FACE_deploy.prototxt'
    #param_fn = model_path + 'VGG_FACE.caffemodel'
    
    
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    
    open('tmp.prototxt', 'w').write(str(model))
    
    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB  
    
    #print(img.format, img.size, img.mode) 
    
    img = (PIL.Image.open(img_name))  
    #img = img.convert('RGB')
    
    img = np.array(img)
    
    
    blobs = []

    opt_layers = open("layers_list.txt","w")
    opt_layers.write("Model:"+param_fn)
    
    for items in net.params:
        blobs.append(items)
        opt_layers.write(items+"\n")
    
    for index,layer in enumerate(blobs):
        if print_only:
            print index,layer
        else:
            frame = deepdream(net, img, end=layer)            
            PIL.Image.fromarray(np.uint8(frame)).save("frames\\"+"%4d-%s.jpg"%(index,layer.replace('/','-')))
    
    print "Done!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepDreamLayers')
    parser.add_argument('-inp', '--input', help='Input file (PNG/JPG)', required=True)
    parser.add_argument('-deploy_file', '--deploy_file', help='Directory of deploy.prototxt', required=True)
    parser.add_argument('-caffe_model', '--caffe_model', help='Directory of .caffemodel', required=True)
    parser.add_argument('-p', '--print_only', help='Print only layer names. Default: 1.If set to 0, will also output the result in a folder - frames',type=int, required=False,default = 1)
    parser.add_argument('-oct', '--octaves', help='Octaves. Default: 4', type=int, required=False,default = 4)
    parser.add_argument('-oct_s', '--octave_scale', help='Octave Scale. Default: 1.5', type=float, required=False, default = 1.4)
    parser.add_argument('-itr', '--iterations', help='Iterations. Default: 10', type=int, required=False, default = 10)
    parser.add_argument('-j', '--jitter', help='Jitter. Default: 32', type=int, required=False, default = 32)
    parser.add_argument('-s', '--step_size', help='Step Size. Default: 1.5', type=float, required=False, default = 1.5)
    parser.add_argument('-gpu', '--gpu', help='Use GPU or CPU.', type=int, required=False,default = 0)
    parser.add_argument('-scale_coef', '--scale_coefficient', help='Scale coefficient for go_deeper mode.', type=float, required=False,default = 0.05)
    
    
    
    
    args = parser.parse_args()
        
    main(args.input,args.deploy_file,args.caffe_model,args.print_only,args.octaves, args.octave_scale, args.iterations, args.jitter,
             args.step_size,args.gpu,args.scale_coefficient) 
