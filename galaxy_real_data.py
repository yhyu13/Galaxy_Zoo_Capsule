"""
Special Note: Small Galaxy Zoo images are written in .jpg extension
              Opencv images don't support .jpg, so they are written in .png extension

"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

GALAXY_IMG_FOLDER =  './v0828new/'
GALAXY_IMG_ID_FILE = './v0828overlap-id.txt'
GALAXY_TRAIN_IMG_ID_FILE = './v0828overlap-train-id.txt'
GALAXY_TEST_IMG_ID_FILE = './v0828overlap-test-id.txt'
GALAXY_IMG_LABEL_FILE = './v0828overlap-catalog.txt'
try:
    DF_IMG_ID = pd.read_csv(GALAXY_IMG_ID_FILE, names=['ID'])
    LIST_IMG_ID = DF_IMG_ID.values.flatten()
except FileNotFoundError:
    print("IMG ID File does not found.")
    
try:
    DF_TRAIN_IMG_ID = pd.read_csv(GALAXY_TRAIN_IMG_ID_FILE, names=['ID'])
    LIST_TRAIN_IMG_ID = DF_TRAIN_IMG_ID.values.flatten()
    
    DF_TEST_IMG_ID = pd.read_csv(GALAXY_TEST_IMG_ID_FILE, names=['ID'])
    LIST_TEST_IMG_ID = DF_TEST_IMG_ID.values.flatten()
except FileNotFoundError:
    print("IMG ID File does not found.")
    
    
def train_test_split(x):
    np.random.shuffle(x)
    threshold = int(len(x)*.8)
    training, test = x[:threshold], x[threshold:]
    with open('./v0828overlap-train-id.txt','w+') as f:
        for ID in training:
            f.write(ID + '\n')
    with open('./v0828overlap-test-id.txt','w+') as f:
        for ID in test:
            f.write(ID + '\n')
            
###
#  Main Function to generate training/testing dataset
###
def generate_train_img_label(is_train=True,additional_label=1):

    if is_train:
        iters = len(LIST_TRAIN_IMG_ID)
    else:
        iters = len(LIST_TEST_IMG_ID)
    
    for i in iters:
        img,label = get_img_label(i, is_train, additional_label)


###
#  Get the source image and label for the ith image in the train/test text file
#  Can have additional label for non-overlapping galaxies (ordered by area)
###      
def get_img_label(index, is_train=True, additional_label=1):

    if is_train:
        list_ID = LIST_TRAIN_IMG_ID
    else:
        list_ID = LIST_TEST_IMG_ID
        
    filename = GALAXY_IMG_FOLDER + list_ID[index]
    image = cv2.imread(filename,1)
    labels = get_bounding_box(image,additional_label)
    return image, labels

# create bounding box for the largest contour in the image
def get_bounding_box(image, additional_label=1,show_img=False):
    img_h, img_w = image.shape[:-1]
    img_h_2, img_w_2 = img_h//2, img_w//2

    x,y,h,w = 0,0,0,0
    label = [0,0,0,0,0]
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(im_bw, 63, 255, 0)
    _ , contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    data = [None] * len(contours)
    
    for i in range(len(contours)):
        x,y,h,w = cv2.boundingRect(contours[i])
        area = h * w
        
        if .90* img_h_2 < x + h/2 < 1.1* img_h_2 and .9* img_w_2 < y + w/2 < 1.1* img_w_2:
            data[i] = [area,1,x/img_h,y/img_w,h/img_h,w/img_w]
        else:
            data[i] = [area,0,x/img_h,y/img_w,h/img_h,w/img_w]
        
    data_overlapping = np.array([points for points in data if points[1] == 1])
    data_overlapping = data_overlapping[data_overlapping[:,0].argsort()]
    label = np.asarray([data_overlapping[-1,1:]])
             
    if show_img:
        x,y,h,w = label[1:]
        cv2.rectangle(image, (x,y), (x+h,y+w),(255,0,0),1)  
        plt.figure()
        plt.imshow(image.astype(int))
        plt.show()
    
    if additional_label >= 1:
        data_nonoverlapping = np.array([points for points in data if points[1] == 0])
        data_nonoverlapping = data_nonoverlapping[data_nonoverlapping[:,0].argsort()]
        label = np.vstack((label,data_nonoverlapping[-additional_label:,1:]))
    return label

###
#  Dihedral transformation of bbox (should be in range [0,1] for each label)
###
def flip_axis_1(x,y,h,w):
    return (1-(x+h)),y,h,w
    
def flip_axis_0(x,y,h,w):
    return x,(1-(y+w)),h,w

def rot90(x,y,h,w):
    return y,(1-(x+h)),w,h
    
def dihedral_transform(img_0,labels_0,action:0):
    img = img_0.copy()
    labels = labels_0.copy()
    
    if action == 0:
        pass
    elif action == 1:
        img = np.flip(img,axis=0)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_0(*labels[i,1:])
    elif action == 2:
        img = np.flip(img,axis=1)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_1(*labels[i,1:])
    elif action == 3:
        img = np.flip(np.flip(img,axis=0),axis=1)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_0(*labels[i,1:])
            labels[i,1:] = flip_axis_1(*labels[i,1:])
    elif action == 4:
        img = np.flip(np.flip(img,axis=1),axis=0)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_1(*labels[i,1:])
            labels[i,1:] = flip_axis_0(*labels[i,1:])
    elif action == 5:
        img = np.rot90(img,k=1)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
    elif action == 6:
        img = np.rot90(img,k=2)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
    elif action == 7:
        img = np.rot90(img,k=3)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
    else:
        print("Dihedral group action should be in range [0,7]")
    
    return img, labels
        
    

def augmentationV2(x,max_offset=50):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:], offsets
    
        
def multigalaxy_generate_sample_alexnetV3(is_shift_ag=True, is_train = True, noFN=True):
    max_offset = int(is_shift_ag) * OFF_SET
    coins = np.random.uniform(size=iters)
    if is_train:
        get_batch_func = galaxy_train_next_batch
    else:
        get_batch_func = galaxy_test_next_batch
    for i in range(iters):
        batch1 = get_batch_func(batch_size, get_img_labelV2)
        batch2 = get_batch_func(batch_size, get_img_labelV2)
        
        images1, offset1 = augmentationV2(batch1[0],max_offset)
        images2, offset2 = augmentationV2(batch2[0],max_offset)
        y1,y2 = batch1[1][:,:2],batch2[1][:,:2]
        bb1,bb2 = batch1[1][:,2:],batch2[1][:,2:]
        #print(bb1,offset1,bb2,offset2)
        bb1[:,0], bb1[:,1] = bb1[:,0]+offset1[0]-max_offset, bb1[:,1]+offset1[1]-max_offset
        bb2[:,0], bb2[:,1] = bb2[:,0]+offset2[0]-max_offset, bb2[:,1]+offset2[1]-max_offset
        # noFN means not including any False-Negative samples
        images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
        y0 = np.logical_or(y1,y2).astype(np.float32)
        bb0 = np.zeros_like(bb1)
        bb0[:,0] = min(bb1[:,0],bb2[:,0])
        bb0[:,1] = min(bb1[:,1],bb2[:,1])
        bb0[:,2] = max(bb1[:,0]+bb1[:,2],bb2[:,0]+bb2[:,2]) - bb0[:,0]
        bb0[:,3] = max(bb1[:,1]+bb1[:,3],bb2[:,1]+bb2[:,3]) - bb0[:,1]
        
        batch3 = get_batch_func(batch_size, get_img_labelV2)
        images3 = batch3[0]
        
        images0 = np.zeros_like(images)
        size = images0.shape[1]
        
        resize2 = np.random.randint(low=size//2,high=size)
        x2,y2 = np.random.randint(size-resize2), np.random.randint(size-resize2)
        images3 = cv2.resize(images3[0],(resize2,resize2))
        
        resize1 = np.random.randint(low=size//5,high=size//2)
        x1,y1 = np.random.randint(size-resize1), np.random.randint(size-resize1)
        images = cv2.resize(images[0],(resize1,resize1))
        
        
        
        images0[:,x2:x2+resize2,y2:y2+resize2] += images3
        images0[:,x1:x1+resize1,y1:y1+resize1] += images
        
        bb0 = bb0 * (resize1 / size)
        bb0[:,0] += x1
        bb0[:,1] += y1 
            
        yield images0, bb0, y0, bb1, y1, bb2,y2


if __name__=='__main__':
    # funtionality test
    
    img, labels = get_img_label(index=0, is_train=True, additional_label=2)
    print(labels)
    colors = {0:(255,0,0),1:(0,255,0)}
    img_0 = img.copy()
    for label in labels:
        bb0 = label[1:].copy()
        bb0[0] *= img_0.shape[0]
        bb0[1] *= img_0.shape[1]
        bb0[2] *= img_0.shape[0]
        bb0[3] *= img_0.shape[1]
        bb0 = [int(e) for e in bb0]
        cv2.rectangle(img_0, (bb0[0],bb0[1]), (bb0[0]+bb0[2],bb0[1]+bb0[3]),colors[0], 0)
        
    plt.figure()
    plt.imshow(img_0.astype(int))
    plt.show()
    
    for action in range(8):
        img_0, labels_0 = dihedral_transform(img,labels,action=action)
        print(labels_0)
        img_0 = img_0.copy()
        for label in labels_0:
            bb0 = label[1:].copy()
            bb0[0] *= img_0.shape[0]
            bb0[1] *= img_0.shape[1]
            bb0[2] *= img_0.shape[0]
            bb0[3] *= img_0.shape[1]
            bb0 = [int(e) for e in bb0]
            cv2.rectangle(img_0, (bb0[0],bb0[1]), (bb0[0]+bb0[2],bb0[1]+bb0[3]),colors[0], 0)
            
        plt.figure()
        plt.imshow(img_0.astype(int))
        plt.show()
