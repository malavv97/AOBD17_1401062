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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()
def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

'''
train_data = 1

x_t = mnist.train.images[:55000,:]
y_t = mnist.train.labels[:55000]

x_train = read('car_cifar_5000.file')

unique, counts = np.unique(y_t, return_counts=True)
count_freq = dict(zip(unique, counts))
no_train_data = count_freq[train_data]

print 'No of training data for letter ',str(train_data),' :- ',no_train_data

x_train = np.zeros([0,784])

j=0
i=0
while i<no_train_data:  
    if y_t[j]==train_data:
        x_train = np.append(x_train,x_t[j].reshape([1,784]),0)
        i+=1
    j+=1
x_train = x_train.astype('float32')
'''

no_train_data = 5000
x_train = read('generated_pca/face.file')

print x_train.shape

no_pc = 512
itr = 35
mse = 10**(-4)

x_train = x_train.transpose()
[W,sigma_square,Xn,t_mean,M] = em_ppca(x_train,no_pc,itr,mse)

W = W.astype('float32')
t_mean = t_mean.astype('float32')
write(W,str(no_train_data)+'_face_W.file')
write(t_mean,str(no_train_data)+'_face_t_mean.file')
'''
img=mpimg.imread('cifar_generated.png')
plt.imshow(img,cmap="gray_r")
plt.show()
'''
