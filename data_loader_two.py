## stable version

## using 4spaces

## TECH - 9 volunteers => 9 * 4 * 16 - ivim images
## +1 volunteers in tech with 3 scans: add that too 

## PMRI 16 b-vlas for baseline scans, 10 b-vals for hq scans (test and retest)
## 

## PMRI series for train 
## PMRI series for test

import tensorflow as tf 
#import tensorflow_addons as tfa
import matplotlib.pylab as plt
import pydicom as dicom
import nibabel as nib
import numpy as np
import cv2
import glob
import os
import fnmatch

## Paths as global variable??

def split_image(img):
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    left_image = img[:, :width_cutoff]
    right_image = img[:, width_cutoff:]
    return left_image, right_image


def flipped_image(image):
    flipped_around_xaxis = cv2.flip(roi_sample, 0) 
    flipped_around_yaxis = cv2.flip(roi_sample, 1)
    return flipped_around_xaxis, flipped_around_yaxis 

"""
def rotate_images(image):
    rotated_images = []
    #image = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
    image = tf.convert_to_tensor(image, tf.float32)
    factors = np.linspace(0,2,25)
    factors = np.delete(factors, 0) # remove 0 at 0 index to avoid division from 0
    for factor in factors:
        rotate = tfa.image.rotate(image, tf.constant(2*np.pi/factor, dtype=tf.float32))
        result = rotate.numpy()
        rotated_images.append(result)

    rotated_images = np.array(rotated_images)
    dims = rotated_images.shape
    rotated_images = np.reshape(rotated_images, (dims[0],dims[1],dims[2]))
    return rotated_images

"""


def rotate_images(image):
    rotated_images = []
    angles = np.arange(0, 375, 15)
    for angle in angles:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        rotated_images.append(result)

    return rotated_images


"""
def load_pmri_data(scans_path = '', roi_path = '', img_split = True, img_normalization = True, 
        data_augmentation = True, three_channel = False, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    pass
"""

def load_pmri_data(set='train', img_split=True, img_normalization=True, 
        data_augmentation=True, three_channel=False, IMG_WIDTH=256, IMG_HEIGHT=256):

    if(set == 'train' or set == 'training'):
        scans_path = '/home/besanhalwa/Eshan/data/SegData/PMRI_Dataset/train_images'
        roi_path = '/home/besanhalwa/Eshan/data/SegData/PMRI_Dataset/train_rois'
    if(set == 'val' or set == 'validation' or set == 'test' or set == 'testing'):
        scans_path = '/home/besanhalwa/Eshan/data/SegData/PMRI_Dataset/test_images'
        roi_path = '/home/besanhalwa/Eshan/data/SegData/PMRI_Dataset/test_rois' 

    scans = glob.glob('{}/*/*TRACE*'.format(scans_path))
    scans.sort()
    
    roi_list = glob.glob('{}/*/*roi*'.format(roi_path))
    roi_list.sort()

    training_images = np.zeros((1, IMG_WIDTH, IMG_HEIGHT))
    training_labels = np.zeros((1, IMG_WIDTH, IMG_HEIGHT))

    for scan, roi in zip(scans, roi_list):
        ivim_images = glob.glob('{}/*'.format(scan))
        ivim_images.sort()

        for ivim_image in ivim_images:
            image_ = dicom.dcmread(ivim_image)
            image = image_.pixel_array
            image_dims = image.shape
            reshaped_image = np.reshape(image,(image_dims[0],image_dims[1],1))    ## make match dims as hight width, channel 

            # input_image =  np.concatenate((resized_image, resized_image, resized_image), axis=-1)
            # three_channel_ivim = input_image[0]
            # single_channel_ivim = three_channel_ivim[:, :, 0]
            
            #input_image = cv2.merge((resized_image,resized_image,resized_image))
            #input_image = np.reshape(input_image,(1, IMG_WIDTH, IMG_HEIGHT, 3))
    
    
            if (img_split):
                split_images = split_image(reshaped_image)
                
                for half_image in split_images:
                    reshaped_image = cv2.resize(half_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
                    reshaped_image = np.reshape(reshaped_image, (IMG_WIDTH, IMG_HEIGHT, 1))
                    if(data_augmentation):
                        rotated_images_ = rotate_images(reshaped_image)
                        training_images = np.append(training_images, rotated_images_,axis=0)
                    else:
                        reshaped_image = np.reshape(reshaped_image, (1, IMG_WIDTH, IMG_HEIGHT))
                        training_images = np.append(training_images, reshaped_image, axis=0)
            
            roi_data = nib.load(roi)           
            roi_image = roi_data.get_fdata()
            roi_dims = roi_image.shape
            reshaped_roi = np.reshape(roi_image,(roi_dims[0],roi_dims[1],1))
            rotate_roi = cv2.rotate(reshaped_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            reshaped_roi = np.reshape(rotate_roi,(roi_dims[1],roi_dims[0],1))
    
            if (img_split):
                split_rois = split_image(reshaped_roi)
                for split_roi in split_rois:
                    reshaped_roi = cv2.resize(split_roi, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
                    reshaped_roi = np.reshape(reshaped_roi,(IMG_WIDTH, IMG_HEIGHT, 1))
                    if(data_augmentation):
                        rotated_rois_ = rotate_images(reshaped_roi)
                        training_labels = np.append(training_labels, rotated_rois_, axis=0)
                    else:
                        reshaped_roi = np.reshape(reshaped_roi, (1, IMG_WIDTH, IMG_HEIGHT))
                        training_labels = np.append(training_labels, reshaped_roi, axis=0)
    training_images = np.delete(training_images, obj=0,  axis=0)
    training_labels = np.delete(training_labels, obj=0,  axis=0)

    print('PMRI dataset image shape {}'.format(training_images.shape))
    print('PMRI dataset roi shape {}'.format(training_labels.shape))

    return training_images, training_labels


def load_tech_data(scans_path = '/home/besanhalwa/Eshan/data/TestData/TECH/DICOM/Volunteer/*/*', 
        roi_path = '/home/besanhalwa/Eshan/data/TestData/NYH/', img_split = True, img_normalization = True,
        data_augmentation = True, three_channel = False, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    scans = glob.glob('/home/besanhalwa/Eshan/data/TestData/TECH/DICOM/Volunteer/*/*')
    scans.sort()

    roi_list = []

    nii_files = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(roi_path)
        for f in fnmatch.filter(files, '*.nii')]

    for nii_file in nii_files:
        nii_ = nib.load(nii_file)
        nii_image = nii_.get_fdata()
        dims = nii_image.shape

        if(dims[0] == 126 and dims[1] == 192 and 'roi' in nii_file.lower()):
            roi_list.append(nii_file)
        elif(dims[0] == 192 and dims[1] == 126 and 'roi' in nii_file.lower()):
            roi_list.append(nii_file)

    print('done')
    roi_list.sort()
    temp_idx = 4 #by reading from the list
    tmp_TECH024_ROI_roitoIVIM = roi_list.pop(temp_idx)

    training_images = np.zeros((1, IMG_WIDTH, IMG_HEIGHT))
    training_labels = np.zeros((1, IMG_WIDTH, IMG_HEIGHT))

    for scan, roi in zip(scans, roi_list):
        ivim_images = glob.glob('{}/*'.format(scan))
        ivim_images.sort()

        ## load ivim image 
        for ivim_image in ivim_images:
            image_ = dicom.dcmread(ivim_image)
            image = image_.pixel_array
            image_dims = image.shape
            reshaped_image = np.reshape(image,(image_dims[0],image_dims[1],1))    ## make match dims as hight width, channel 

            # print('training images shape {}'.format(training_images.shape))
            # print('reshaped_image  shape {}'.format(reshaped_image.shape))

            if (img_split):
                split_images = split_image(reshaped_image)

                for half_image in split_images:
                    
                    reshaped_image = cv2.resize(half_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
                    reshaped_image = np.reshape(reshaped_image, (IMG_WIDTH, IMG_HEIGHT, 1))
                    if(data_augmentation):

                        rotated_images_ = rotate_images(reshaped_image)

                        #print('rotated_image shape {}'.format(rotated_images_.shape))
                        #print('training_images shape {}'.format(training_images.shape))

                        training_images = np.append(training_images, rotated_images_,axis=0)
                        # print('shape after augmentation{}'.format(rotated_images_.shape)) ---  correct 
                    else:
                        reshaped_image = np.reshape(reshaped_image, (1, IMG_WIDTH, IMG_HEIGHT))
                        training_images = np.append(training_images, reshaped_image, axis=0)
            
            
            roi_data = nib.load(roi)
            roi_image = roi_data.get_fdata()
            roi_dims = roi_image.shape
            reshaped_roi = np.reshape(roi_image,(roi_dims[0],roi_dims[1],1))
            rotate_roi = cv2.rotate(reshaped_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            reshaped_roi = np.reshape(rotate_roi,(roi_dims[1],roi_dims[0],1))

            if (img_split):
                split_rois = split_image(reshaped_roi)
                for split_roi in split_rois:
                    reshaped_roi = cv2.resize(split_roi, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
                    reshaped_roi = np.reshape(reshaped_roi,(IMG_WIDTH, IMG_HEIGHT, 1))
                    if(data_augmentation):
                        rotated_rois_ = rotate_images(reshaped_roi)
                        training_labels = np.append(training_labels, rotated_rois_, axis=0)
                    else:
                        reshaped_roi = np.reshape(reshaped_roi, (1, IMG_WIDTH, IMG_HEIGHT))
                        training_labels = np.append(training_labels, reshaped_roi, axis=0)


    #training_images = np.array(training_images)
    #training_labels = np.array(training_labels)
    if (three_channel):
        pass
        #training_images =  np.concatenate((training_images, training_images, training_images), axis=-1)
        #training_labels =  np.concatenate((training_labels, training_labels, training_labels), axis=-1)

    training_images = np.delete(training_images, obj=0,  axis=0)
    training_labels = np.delete(training_labels, obj=0,  axis=0)

    print('Tech dataset image shape {}'.format(training_images.shape))
    print('Tech dataset roi shape {}'.format(training_labels.shape))


    return training_images, training_labels


def load_pmri_ten_rois(path = '', img_split = True, img_normalization = True, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    pass


def load_tech_ten_rois(path = '', img_split = True, img_normalization = True, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    pass



## this does not split the image
def load_train_image_sample(IMG_WIDTH=256, IMG_HEIGHT=256):
    example_target_xx = nib.load('/home/besanhalwa/Eshan/data/TestData/NYH/TECH024/TECH024_IVIM_baseline/TECH024_ROI_roitoMPRtoIVIM_baseline.nii')
    example_target = example_target_xx.get_fdata()
    example_target_dims = example_target.shape
    example_target = np.reshape(example_target,(example_target_dims[0],example_target_dims[1],1))
    example_target = cv2.rotate(example_target, cv2.ROTATE_90_COUNTERCLOCKWISE)
    example_target = np.reshape(example_target,(example_target_dims[1],example_target_dims[0],1))
    example_target = cv2.resize(example_target, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
    example_target = np.reshape(example_target,(IMG_WIDTH, IMG_HEIGHT, 1))
    #example_target =  np.concatenate((example_target, example_target, example_target), axis=-1)


    example_input_xx = dicom.dcmread("/home/besanhalwa/Eshan/data/TestData/TECH/DICOM/Volunteer/TECH024/1IVIM_16b_baseline_TRACEW_0003/MR.1.3.12.2.1107.5.2.38.51020.2022032413292747215713224")
    example_input = example_input_xx.pixel_array
    example_input_dims = example_input.shape
    example_input = np.reshape(example_input,(example_input_dims[0],example_input_dims[1],1))
    example_input = cv2.resize(example_input, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)
    example_input = np.reshape(example_input, (IMG_WIDTH, IMG_HEIGHT, 1))
    #example_input =  np.concatenate((example_input, example_input, example_input), axis=-1)

    print(example_input.shape)
    print(example_target.shape)
    
    return example_input, example_target

def load_test_image_sample():
    pass

def load_volunteer_dicoms():
    pass 

def load_volunteer():
    pass 

def load_npy_input_lable():
    pass

def load_training_data(img_split = True, img_normalization = True, data_augmentation = True, three_channel = False, 
        IMG_WIDTH = 256, IMG_HEIGHT = 256):
    training_images = 0
    training_labels = 0

def load_test_data(img_split = True, img_normalization = True, data_augmentation = False, three_channel = False, 
        IMG_WIDTH = 256, IMG_HEIGHT = 256):
    test_images = 0
    test_labels = 0

def load_training_data_npy(path='/home/besanhalwa/Eshan/data/npy'):
    images = np.append(np.load('{}/tech_training_images_aug.npy'.format(path)), 
                       np.load('{}/pmri_training_images_aug.npy'.format(path)), axis=0)
    labels = np.append(np.load('{}/tech_training_labels_aug.npy'.format(path)), 
                       np.load('{}/pmri_training_labels_aug.npy'.format(path)), axis=0)
    return images, labels


def load_test_data_npy(path='/home/besanhalwa/Eshan/data/npy', data_augmentation = False):
    if(data_augmentation == False):
        test_images = np.load('{}/pmri_test_images_no_aug.npy'.format(path))
        test_labels = np.load('{}/pmri_test_labels_no_aug.npy'.format(path))
    else:
        test_image = np.load('{}/pmri_test_images_aug.npy'.format(path))
        test_labels = np.load('{}/pmri_test_labels_aug.npy'.format(path))
    return test_images, test_labels


"""
converting single channel data into three channel 

arr1 = np.zeros((1,256,256))  ## image_set 1
arr2 = np.zeros((1,256,256))  ## image_set 2
arr3 = np.append(arr1, arr2, axis=0)

arr3.shape ## (2, 256, 256)

arr3 = arr3.reshape(*arr3.shape[:], -1)
arr3.shape  ## (2, 256, 256, 1)

three =  np.concatenate((arr3, arr3, arr3), axis=-1)
three.shape ## (2, 256, 256, 3)

"""
