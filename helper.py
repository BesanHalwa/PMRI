import tensorflow as tf
import numpy as np
import cv2 
import matplotlib.pyplot as plt

import glob
import matplotlib.pylab as plt
import pydicom as dicom
import nibabel as nib

import os

def plot(scan, roi, bval, manual_roi):
    #mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(scan, cmap='gray')
    plt.title('TRACE_W; b_val='+str(bval))


    fig.add_subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(roi, cmap='gray')
    plt.title('Predicted ROI')

    fig.add_subplot(1, 3, 3) 
    plt.axis('off')
    plt.imshow(scan, cmap='gray', alpha=1, interpolation='none') 
    plt.imshow(roi, 'jet', alpha=0.6, interpolation='none')
    plt.title('Overlayed ROI')

    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(2, 3, 1)
    plt.axis('off')
    plt.imshow(scan, cmap='gray')
    plt.title('TRACE_W; b_val='+str(bval))


    fig.add_subplot(2, 3, 2)
    plt.axis('off')
    plt.imshow(manual_roi, cmap='gray')
    plt.title('Manual ROI')

    fig.add_subplot(2, 3, 3) 
    plt.axis('off')
    plt.imshow(scan, cmap='gray', alpha=1, interpolation='none') 
    plt.imshow(roi, 'jet', alpha=0.6, interpolation='none')
    plt.title('Overlayed ROI')
    
    plt.show()

def increase_class_id(img):
    rows, coloumns = img.shape
    for row in range(rows):
        for coloumn in range(coloumns):
            if img[row,coloumn] > 0:
                img[row,coloumn] = img[row,coloumn] + 5
    return img

def hot_encoded_to_single_channel(mask):

  # Check if the input has six channels (expected for hot-encoded mask)
  if mask.shape[-1] != 6:
    raise ValueError("Expected six-channel hot-encoded mask.")

  # Get the class with the highest probability for each pixel (argmax)
  dominant_class = np.argmax(mask, axis=-1)

  return dominant_class

# Data preprocessing
def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    return input_image

def load_nifty(path):
    nifti_file = nib.load(path)
    image_data = nifti_file.get_fdata()
    image_data = np.squeeze(image_data)
    image_data= cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_data