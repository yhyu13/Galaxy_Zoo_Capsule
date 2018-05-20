import numpy as np
import pandas as pd
import cv2

GALAXY_TRAIN_FILE = './data/train_simple.txt'
GALAXY_TEST_FILE = './data/test_simple.txt'
GALAXY_ORIG_FOLDER = './data/images_training_rev1/'

DF_TRAIN = pd.read_csv(GALAXY_TRAIN_FILE,names=['ID','label'])
DF_TEST = pd.read_csv(GALAXY_TEST_FILE,names=['ID','label'])

OFF_SET = 20

def galaxy_train_next_batch(batch_size):
    df_samples = DF_TRAIN.sample(batch_size)
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

def galaxy_test_next_batch(batch_size):
    df_samples = DF_TEST.sample(batch_size)
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

def augmentation(x,max_offset=2):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:]

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

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from time import sleep
    mm_train_iter = multigalaxy_train_iter(iters=10)
    for img_mmtrain,_ in mm_train_iter:
        img_mmtrain_sample = img_mmtrain[0,:,:,0]
        plt.imshow(img_mmtrain_sample)
        plt.show()
