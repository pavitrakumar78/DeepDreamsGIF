# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 14:28:25 2015

@author: Pavitrakumar
"""

# imports and basic notebook setup
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


def make_step(net, step_size=1.5, end='inception_4c/output', 
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
              end='inception_4c/output', jitter = 32,step_size=1.5):
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


     
def main(img_name,octaves=4,octave_scale=1.4,iterations=10,jitter=32,step_size=1.5,layer='inception_4c/output',gpu=0,go_deeper=50,scale_coefficient=0.05):
    path = os.getcwd()
    
    gif_mode = 0
    
    if not os.path.exists(path+"\\gifsicle.exe"):
        print "Can't process GIFs, gifsicle.exe not found!"
        exit(1)
    if os.system("mkdir dreams"): #create a remporary file for image processing/storing
        print "temp. output folder already exists, deleting exisisting one.."
        os.system("rm -r dreams")
        os.system("mkdir dreams")
    #Use gifsicle to get frames from the gif and put it in an temporary output folder
    if not os.path.exists(img_name):
        print "No input file found!"
        exit(1)
    if img_name[-3:]=='gif':
        gif_mode = 1
        os.system("gifsicle --explode -U "+img_name+" --output frame")
        os.system("ren frame.* frame.*.jpg")
    
        if os.system("move *.jpg dreams"):
            print "Can't move files!"
            exit(1)
#    else,
        #go-deeper mode | param required or - default is 50
        
    
    if gpu:
        caffe.set_mode_gpu()
    else:
        print "You are using CPU mode, this might take some time"
       
    
    model_path = './caffe/models/bvlc_googlenet/' # substitute your path here
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'
    
    """
    Other models : ( need to change default end param if you are going to use this )
    
    model_path = './caffe/models/vgg_face_caffe/' 
    net_fn   = model_path + 'VGG_FACE_deploy.prototxt'
    param_fn = model_path + 'VGG_FACE.caffemodel'
    #(example end params: 'conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3','pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5')
    
    model_path = './caffe/models/finetune_flickr_style/' 
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'finetune_flickr_style.caffemodel'
    #(example end params: 'conv1','pool1','norm1','conv2','pool2','norm2','conv3','conv4','conv5','pool5')
    """
    
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    
    
    if gif_mode:
        cnt = 0
        dr = path+"\\dreams"
        total = len(os.listdir(dr))-1
        for i in os.listdir(dr):
            if i.endswith(".jpg"):
                frame_name =  dr+"\\"+i
                img = (PIL.Image.open(frame_name))
                img = img.convert('RGB')
                #print(img.format, img.size, img.mode) 
                dream_img = deepdream(net, np.array(img),iter_n=iterations,octave_n=octaves,octave_scale=octave_scale,end=layer,jitter=jitter,step_size=step_size)
                dream_img = img_fromarray(np.uint8(dream_img)).convert('P', palette=PIL.Image.ADAPTIVE)
                dream_img.save(dr+"\\dreamimg"+str(i)+".gif")
                os.system("rm dreams\\"+i)
                cnt+=1
                print str(cnt)+" frames completed out of "+str(total)
    else: #go-deeper mode, takes in a single jpg image and dreates dreams of itself.
        img = (PIL.Image.open(path+"\\"+img_name))
        img = img.convert('RGB')
        img = np.array(img)
        frame = img
        h, w = frame.shape[:2]
        s = scale_coefficient # scale coefficient

        for i in xrange(go_deeper):
            frame = deepdream(net, frame)
            PIL.Image.fromarray(np.uint8(frame)).convert('P', palette=PIL.Image.ADAPTIVE).save(path+"\\"+"dreams\\%04d.gif"%i)
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
            print str(i)+" frames completed out of "+str(go_deeper)
            


    os.system("gifsicle --loop=0 dreams/*.gif > "+img_name[:-4]+"-dream.gif");
    os.system("rm -r dreams");
    print "File saved as "+img_name[:-4]+"-dream.gif"
    print "Done!"
    exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepDreamGIF')
    parser.add_argument('-inp', '--input', help='Input file (GIF/PNG/JPG)', required=True)
    parser.add_argument('-oct', '--octaves', help='Octaves. Default: 4', type=int, required=False,default = 4)
    parser.add_argument('-oct_s', '--octave_scale', help='Octave Scale. Default: 1.5', type=float, required=False, default = 1.4)
    parser.add_argument('-itr', '--iterations', help='Iterations. Default: 10', type=int, required=False, default = 10)
    parser.add_argument('-j', '--jitter', help='Jitter. Default: 32', type=int, required=False, default = 32)
    parser.add_argument('-ss', '--step_size', help='Step Size. Default: 1.5', type=float, required=False, default = 1.5)
    parser.add_argument('-l', '--layer', help='Layer to use. Default: inception_4c/output. Suggested Layers: inception_3b/5x5_reduce,inception_4e/pool_proj', type=str,required=False,  default = "inception_4c/output")
    parser.add_argument('-gpu', '--gpu', help='Use GPU or CPU.', type=int, required=False,default = 0)
    parser.add_argument('-dpr', '--go_deeper', help='Use single frame and feed result (dream) of it to itself.This is the default option if the input file is jpg/png.', type=int, required=False,default = 50)
    parser.add_argument('-scale_co', '--scale_coefficient', help='Scale coefficient for go_deeper mode.', type=float, required=False,default = 0.05)
 
    
    
    args = parser.parse_args()
        
    main(args.input,args.octaves, args.octave_scale, args.iterations, args.jitter,
             args.step_size, args.layer,args.gpu,args.go_deeper,args.scale_coefficient) 

    
    
    
    
    
    
