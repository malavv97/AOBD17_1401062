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

def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()
def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

dloss_t = read('Results/Dloss_1000_traditional.file')
dloss_o = read('Results/Dloss_1000_proposed_gm.file')

plt.plot(dloss_t,'red')
plt.plot(dloss_o,'green')

plt.legend(['Dloss in traditional model', 'Dloss in Proposed model'], loc='upper right')

plt.xlabel('Over Iterations')
plt.ylabel('Discriminator loss')
plt.title('Dloss comparison between Traditional and Proposed Generative Model')
plt.show()
