from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    
    #image_pil = Image.fromarray(image_numpy)
    #image_pil.save(image_path)
    image_float = image_numpy.astype(np.float64)
    if (np.max(image_float) - np.min(image_float)) == 0:
        norm = np.zeros_like(image_float)  # or any other appropriate handling
    else:
        norm = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float))
    

    # Convert the normalized image to uint8 format expected by PIL
    norm_uint8 = (norm * 255).round().astype(np.uint8)  # Multiply by 255 and convert to uint8

    # Create a PIL image from the array and save it
    image_pil = Image.fromarray(norm_uint8)
    image_pil.save(image_path)

def save_image(image_numpy, image_path):
    
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
