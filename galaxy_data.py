"""
Special Note: Small Galaxy Zoo images are written in .jpg extension
              Opencv images don't support .jpg, so they are written in .png extension

"""

import numpy as np
import pandas as pd
import cv2
import sys

GALAXY_TRAIN_FILE = './data/train_simple.txt'
GALAXY_TEST_FILE = './data/test_simple.txt'
GALAXY_ORIG_FOLDER = './data/images_training_rev1/'
GALAXY_TRAIN_FOLDER = './data/train_samples/'

DF_TRAIN = pd.read_csv(GALAXY_TRAIN_FILE,names=['ID','label'])
DF_TEST = pd.read_csv(GALAXY_TEST_FILE,names=['ID','label'])

OFF_SET = 50

# get data batch
def get_img_label(df_samples, batch_size):
    images = np.zeros([batch_size, 212, 212, 1])
    labels = np.zeros([batch_size,2])
    images_id = list(df_samples['ID'])
    images_label = list(df_samples['label'])
    for i in range(batch_size):
        filename = GALAXY_ORIG_FOLDER + '%d' % images_id[i] + '.jpg'
        images[i] = np.asarray(cv2.resize(cv2.imread(filename,0),(212,212))).reshape((212,212,1))
        if images_label[i] == 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return images, labels

def get_img_label_alexnet(df_samples, batch_size):
    images = np.zeros([batch_size, 224, 224, 3])
    labels = np.zeros([batch_size,2])
    images_id = list(df_samples['ID'])
    images_label = list(df_samples['label'])
    for i in range(batch_size):
        filename = GALAXY_ORIG_FOLDER + '%d' % images_id[i] + '.jpg'
        images[i] = np.asarray(cv2.resize(cv2.cvtColor(cv2.imread(filename,0),cv2.COLOR_GRAY2RGB),(224,224))).reshape((224,224,3))
        if images_label[i] == 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return images, labels

def galaxy_train_next_batch(batch_size, get_image_label_func=get_img_label):
    df_samples = DF_TRAIN.sample(batch_size)
    return get_image_label_func(df_samples, batch_size)

def galaxy_test_next_batch(batch_size,  get_image_label_func=get_img_label):
    df_samples = DF_TEST.sample(batch_size)
    return get_image_label_func(df_samples, batch_size)

# argument data with shift offsets
def augmentation(x,max_offset=2):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:]

# training & testing generators
def galaxy_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch = galaxy_train_next_batch(batch_size)
        images = batch[0]
        images = np.concatenate([images]*3,axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def galaxy_test_iter(iters=1000,batch_size=32,is_shift_ag=False):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch = galaxy_test_next_batch(batch_size)
        images = batch[0]
        images = np.concatenate([images] * 3, axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def multigalaxy_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch1 = galaxy_train_next_batch(batch_size)
        batch2 = galaxy_train_next_batch(batch_size)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        images = np.clip(np.add(images1,images2).astype(np.float32),0,255)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)

def multigalaxy_test_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch1 = galaxy_test_next_batch(batch_size)
        batch2 = galaxy_test_next_batch(batch_size)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        images = np.clip(np.add(images1,images2).astype(np.float32),0,255)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)


# Code for 224x224x3 input size

GALAXY_TRAIN_FOLDER =  './data/train_samples/'
GALAXY_TRAIN_FILE = './data/train_samples/lens_parameters.txt'
try:
    DF_LABEL = pd.read_csv(GALAXY_TRAIN_FILE, names=['eplliptical','spiral'])
    LIST_LABEL = DF_LABEL.values
except FileNotFoundError:
    print("Training dataset has not created yet!")

# get data batch
def get_img_label_alexnetV2(start_index, batch_size):
    images = np.zeros([batch_size, 224, 224, 3])
    start_index = start_index * batch_size
    err = False
    for i in range(batch_size):
        filename = GALAXY_TRAIN_FOLDER + "img" + '_' + "%07d" % (start_index+i+1) + '.png'
        img = cv2.imread(filename,0)
        if img is not None:
            images[i] = np.asarray(cv2.resize(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),(224,224))).reshape((224,224,3))
        else:
            err = True
    labels = LIST_LABEL[start_index:start_index+batch_size]
    return images, labels, err

def multigalaxy_train_iter_alexnet(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        yield get_img_label_alexnetV2(i, batch_size)

# genreate versatile categories of data
def multigalaxy_generate_sample_alexnet(iters=1000,batch_size=1,is_shift_ag=True, is_train = True):
    max_offset = int(is_shift_ag) * OFF_SET
    coins = np.random.uniform(size=iters)
    if is_train:
        get_batch_func = galaxy_train_next_batch
    else:
        get_batch_func = galaxy_test_next_batch
    for i in range(iters):
        batch1 = get_batch_func(batch_size, get_img_label_alexnet)
        batch2 = get_batch_func(batch_size, get_img_label_alexnet)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        y1,y2 = batch1[1],batch2[1]
        """
        if coins[i] < 0.25:
            images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
            y0 = np.logical_or(y1,y2).astype(np.float32)
        elif 0.25 <= coins[i] < 0.5:
            images = images1
            y0 = y1
        elif 0.5 <= coins[i] < 0.75:
            images = images2
            y0 = y2
        else:
            images = np.asarray([255 * np.random.random((224,224,3))])
            y0 = np.zeros([1,2])
        """
        if coins[i] < 0.5:
            images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
            y0 = np.logical_or(y1,y2).astype(np.float32)
        elif 0.5 <= coins[i] < 0.75:
            images = images1
            y0 = y1
        else:
            images = images2
            y0 = y2
            
        yield images, y0

if __name__=='__main__':
    # funtionality test
    import matplotlib.pyplot as plt
    train = multigalaxy_train_iter_alexnet(10, 1)
    for img, label in train:
        print(label)
        plt.figure()
        plt.imshow(img[0].astype(int))
        plt.show()
