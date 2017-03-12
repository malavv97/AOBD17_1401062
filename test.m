clear all;
clc;

%% Import Image
RGB = imread('.\data\1.jpg');% convert given file into RGB
t = rgb2gray(RGB);% convert RGB file into gray image
t = im2double(t);% convert uint8 value to double for computation

imshow(im2uint8(t));

%% Make Image corrupted
k = 220;

%% Apply PPCA on data corrupted data matrix
[W,sigma_square,Xn,t_mean] = em_ppca(t,k);

%% Apply PPCA with Expectation Maximization Algorithm
[W_,sigma_square_,Xn_,t_mean_] = ppca(t,k);

%% Recovered Image
rec_Image = W*inv(W'*W)*Xn;
disp('Perfect');
rec_Image = rec_Image + (t_mean)*ones(1,size(rec_Image,2));

imshow(im2uint8(rec_Image));