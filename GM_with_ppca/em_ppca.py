import numpy as np
import math as mt
from numpy.linalg import inv
from numpy import linalg as LA

def em_ppca(data,no_pc,max_iter,error):

	# Data  : for which we want to find principal components
	# no_pc : how many principal components use wants
	#		  -> condition no_pc < no of data vectors (columns)
	# max_iter : maximum iteration that user allow to run EM algorithm
	# error : Mean Squared Error that user allows	

	# Take height and width of data matrix
	data_size = data.shape
	height = data_size[0]
	width = data_size[1]

	# Check dimensionality constraint
	if no_pc<1 or no_pc>width:
		print("Oops!  That was no valid number.  Try again...")
		return(0)

	# Find mean value of given matrix
	t_mean = np.zeros((height, 1),dtype='float64')
	for i in range(0,height):
		for j in range(0,width):
			if data[i][j] is not None:
				t_mean[i][0] = t_mean[i][0] + data[i][j]
	t_mean = t_mean/height

	# Normalize Data Matrix
	for i in range(0,height):
		for j in range(0,width):
			if data[i][j] is not None:
				data[i][j] = data[i][j] - t_mean[i][0];
			else:
				data[i][j] = 0;

	# Initialy W and sigma^2 will be randomly selected.
	mu,sigma = 0,1
	print('W and Sigma square are initialized')
	W = np.random.normal(mu, sigma, (height,no_pc))
	sigma_square = np.random.normal(mu,sigma)

	print('EM Algorithm is running')
	for i in range(0,max_iter):
		#Find W transpose
		W_T = W.transpose()
		#Find M = W'W + Sigma^2*I
		M = np.dot(W_T,W) + (sigma_square*np.identity(no_pc))
		# Find Inverse of Matrix
		In_M = inv(M)
		# Expected Xn
		Xn = np.zeros((no_pc, width),dtype='float64')
		Xn_Xn_T = np.zeros((no_pc, no_pc),dtype='float64')

		for j in range(0,width):
			Xn[:,j] = np.dot(np.dot(In_M,W_T),data[:,j])
			Xn_Xn_T = Xn_Xn_T + ((sigma_square*In_M)+(np.dot(Xn[:,j].reshape(len(Xn),1),(Xn[:,j].reshape(len(Xn),1).transpose()))))

		old_W = W

		temp1 = np.zeros((height, no_pc),dtype='float64')
		for j in range(0,width):
			temp1 = temp1 + (np.dot(data[:,j].reshape(height,1),(Xn[:,j].reshape(len(Xn),1).transpose())))

		# Take new value of W
		W = np.dot(temp1,inv(Xn_Xn_T))

		sum11 = 0
		for j in range(0,width):
			temp2 = sigma_square*(In_M) + np.dot(Xn[:,j].reshape(len(Xn),1),(Xn[:,j].reshape(len(Xn),1).transpose()))
			sum11 = sum11 + mt.pow(LA.norm(data[:,j]),2) - 2*np.dot(np.dot(Xn[:,j].reshape(len(Xn),1).transpose(),W_T),data[:,j].reshape(height,1)) + np.trace(np.dot(np.dot(temp2,W_T),W))

		sigma_square = sum11/(height*width)

		mse = ((old_W-W)**2).mean(axis=None)

		if(mse<error):
			break;
		else:
			print 'iter :- ',i+1,' Mean Squared Error :- ',mse

	print('EM is finished')
	M = np.dot(W_T,W) + sigma_square*np.identity(no_pc)
	In_M = inv(M)
	Xn = np.dot(np.dot(In_M,W_T),data)

	return W,sigma_square,Xn,t_mean,M
