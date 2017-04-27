import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

W1 = tf.Variable([1.0], tf.float32)
W2 = tf.Variable([1.0], tf.float32)
b1 = tf.Variable([1.0], tf.float32)
b2 = tf.Variable([1.0], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
#linear_model = (W+x) * (b+y)
model = (W1*x + b1) * (W2*y+b2) 

sess = tf.Session()

init = tf.global_variables_initializer()

z = tf.placeholder(tf.float32)
squared_deltas = tf.square(model - z)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)
a = []
num1 = input('Enter number 1: ')
num2 = input('Enter number 2: ')
temp = num1*num2
	
sess.run(init) # reset values to incorrect defaults.

for i in range(1,21):
	for j in range(50*i):
		sess.run(train, {x:[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5], y:[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10], z:[1,2,3,4,5,6,7,8,9,10,2,4,6,8,10,12,14,16,18,20,3,6,9,12,15,18,21,24,27,30,4,8,12,16,20,24,28,32,36,40,5,10,15,20,25,30,35,40,45,50]})

		
		# print(sess.run([W1, W2, b1, b2]))
	temp1 = sess.run((W1*num1+b1)*(W2*num2 +b2))
	a.append(temp-np.asscalar(temp1[0]))
	
	# print(temp1)
		# for k in range(6,11):
		# 	print("Table of ", k)
		# 	for l in range(1,11):
		# 		temp1 = sess.run((W1*k+b1)*(W2*l +b2))
		# 		print(temp1)
		# 		a.append(k*l - np.asscalar(temp1[0]))


plt.plot(range(50,1050,50), a, 'r')
plt.axis([50,1000,0.0001,5])
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training analysis for tables')

plt.show()


