import tensorflow as tf
from galaxy_CapsNet import *
import cv2
import numpy as np
from galaxy_data import *
from tensorflow.contrib.slim.nets import resnet_v1

batch_size = 32
num_class = 2
is_multi_galaxy = True
is_shift_ag = True
irun = 0
lr = 1e-4
steps = 100000
save_frequence = 2500
decay_frequence = 5000
min_lr = 5e-6

train_iter = multigalaxy_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
test_iter = multigalaxy_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)

tf.reset_default_graph()
tf.set_random_seed(1234)

inputs = tf.placeholder(tf.float32, shape=(None,224,224,3))
target = tf.placeholder(tf.float32, [None, num_class])
dynamic_lr = tf.placeholder(tf.float32, shape=())

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
   net, end_points = resnet_v1.resnet_v1_50(inputs, num_class, is_training=True)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=target))
train = tf.train.AdamOptimizer(learning_rate=dynamic_lr).minimize(loss)

prediction_prob = tf.nn.sigmoid(net)
# let prob1 be [0.4,0.6], prob2 be [0.6,0.8]
# let target1 be [0,1], target2 be [1,0]
# then tf.abs(prediction_prob - target) for the 1st example is 0.8, and it is 1.2 for the 2nd example
# we can then set a ture/false threshold at 1
# then relu will nullify ture predictions as zeros
# finally count nonzero elements, then divided by batch size
error_rate = tf.count_nonzero(tf.nn.relu(tf.reduce_sum(tf.abs(prediction_prob - target), axis=1) - 1),dtype=tf.float32) / tf.shape(target)[0]

tf.summary.scalar('error_rate_on_train_set', error_rate * 100.0)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("./sum",sess.graph)
var_to_save = [var for var in tf.global_variables() if ('Adam' not in var.name) and ('Momentum' not in var.name)]
saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)

sess.run(init)

RESTORE = True
if RESTORE:
    try:
        ckpt = tf.train.get_checkpoint_state('./cpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading model succeeded...')
    except:
        print("Loading model failed or model does't not exist")

for X,Y in train_iter:


    LS, ACC, _ = sess.run([loss, error_rate, train], feed_dict={inputs: X, target: Y, dynamic_lr: lr})

    writer.add_summary(result, irun)

    # print('{},{},{},{},{}'.format(irun, LS, LS_REC, ACC, ACC_TEST))
    print('{}, LS: {:.4f}, ACC:{:.3f}'.format(irun, LS, ACC))

    if (irun+1) % save_frequence == 0:
        saver.save(sess, './cpt/model-{}.ckpt'.format(irun+1))

    if (irun+1) % decay_frequence == 0 and lr > min_lr:
        lr *= 0.5

    irun += 1
irun
"""
irun = 0
for X,Y in test_iter:


    LS, ACC = sess.run([loss, error_rate], feed_dict={inputs: X, target: Y})

    writer.add_summary(result, irun)

    # print('{},{},{},{},{}'.format(irun, LS, LS_REC, ACC, ACC_TEST))
    print('{}, LS: {:.4f}, ACC:{:.3f}'.format(irun, LS, ACC))

    irun += 1
"""
