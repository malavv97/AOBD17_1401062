'''
Generative Model Reference is taken from following link
Ref :- https://github.com/adeshpande3/Generative-Adversarial-Networks
and according to our application, we have modified it
'''

import tensorflow as tf
import numpy as np
import math as mt
import random
import pickle
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.plotly as py
import plotly.tools as tls
from numpy.linalg import inv
from numpy import linalg as LA
from em_ppca import em_ppca
from skimage import color
from math import sqrt
from random import randint

# to write data into file
def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

# to read data into file
def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

# 2d convolution of matrix
def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

# Average pooling
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Discriminator Model
def discriminator(x_image, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    #First Conv and Pool Layers
    W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = avg_pool_2x2(h_conv1)

    #Second Conv and Pool Layers
    W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = avg_pool_2x2(h_conv2)

    #First Fully Connected Layer
    W_fc1 = tf.get_variable('d_wfc1', [8 * 8 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Second Fully Connected Layer
    W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

    #Final Layer
    y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return y_conv

# Generative Model using Principal Components
def generator(z, batch_size, z_dim, W, t_mean,reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    g_dim = 64 #Number of filters of first layer of generator
    c_dim = 1 #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)

    h0 = tf.reshape(z, [batch_size, 2, 2, 25])
    h0 = tf.nn.relu(h0)
    # 1*2*2*25

    output1_shape = [batch_size, 4, 4, g_dim]
    W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
    H_conv1 = tf.nn.relu(H_conv1)
    # 1*4*4*64

    output2_shape = [batch_size, 8, 8, g_dim/2]
    W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
    H_conv2 = tf.nn.relu(H_conv2)
    # 1*8*8*32

    output3_shape = [batch_size, 16, 16, c_dim]
    W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv3 = tf.nn.tanh(H_conv3)
    # 1*16*16*1

    Result = tf.reshape(H_conv3, [batch_size,16*16,1])
    final = tf.zeros([0,32,32,1])
    for i in range(batch_size):
    	c = tf.reshape(tf.add(tf.matmul(W,Result[i]),t_mean),[1,32,32,1])
    	final = tf.concat(0,[final,c])
    return final
    # t = Wx + t_mean


no_train_data = 5000
x_train = read('generated_pca/car_cifar_5000.file')
W = read('generated_pca/5000_cifar_W.file')
t_mean = read('generated_pca/5000_cifar_t_mean.file')

randomNum = random.randint(0,no_train_data-1)
image = x_train[randomNum,:].reshape([32,32])
plt.imshow(image)
plt.show()

sess = tf.Session()
z_dimensions = 100
z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

sample_image = generator(z_test_placeholder, 1, z_dimensions, W, t_mean)
test_z = np.random.normal(-1, 1, [1,z_dimensions])

sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))

my_i = temp.squeeze()
plt.imshow(my_i)
plt.show()

batch_size = 16
tf.reset_default_graph() #Since we changed our batch size (from 1 to 16), we need to reset our Tensorflow graph

sess = tf.Session()
x_placeholder = tf.placeholder(tf.float32, shape = [None,32,32,1]) #Placeholder for input images to the discriminator
z_placeholder = tf.placeholder(tf.float32, shape = [None, z_dimensions]) #Placeholder for input noise vectors to the generator

Dx = discriminator(x_placeholder) #Dx will hold discriminator prediction probabilities for the real MNIST images
Gz = generator(z_placeholder, batch_size, z_dimensions,W,t_mean) #Gz holds the generated images
Dg = discriminator(Gz, reuse=True) #Dg will hold discriminator prediction probabilities for generated images

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dx, tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

sess.run(tf.global_variables_initializer())
iterations = 10000
dloss=[]
for i in range(iterations):
    z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
    real_image_batch = np.empty([batch_size,32,32,1],dtype='float32')

    for j in range(batch_size):
    	real_image_batch[j,:,:,:] = np.reshape(x_train[randint(0,no_train_data-1),:],[1,32,32,1])

    _,dLoss = sess.run([trainerD, d_loss],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch}) #Update the discriminator
    _,gLoss = sess.run([trainerG,g_loss],feed_dict={z_placeholder:z_batch}) #Update the generator
    print 'iter :- ',i,' DLoss :- ',dLoss,' gLoss :- ',gLoss
    dloss.append(np.asscalar(dLoss))


sample_image = generator(z_placeholder, 1, z_dimensions,W,t_mean)
z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i)
plt.show()

plt.plot(dloss, 'r')
plt.xlabel('Over Iterations')
plt.ylabel('Discriminator loss')
plt.title('Discriminator loss in Proposed GM - CIFAR-10 Dataset')
plt.show()

write(dloss,'Results/cifar_5000_dloss.file')
