function [eigval,eigvec] = classical_pca(t,k)

%% find height and width of data matrix
[height,width] = size(t); 

%% Find mean value of [t1 t2 ... tm]
% Mind mean vector of observed data vectors
t_mean = (sum(t(:,1:width)')')/width; 

%% Normalize Data matrix
t = t - (t_mean)*ones(1,width);

%% Find sample co-variance matrix C = AA' where A = [(t1-t_mean) (t2-t_mean) (t3-t_mean) ... (tm-t_mean)];
A = t;

%% Find Eigen value and Eigen vector of C (N^2*N^2) but matrix is to large so it is practically not possible
% Instead of finding eigen values and eigen vectors of A*A', we will find
% eigen value and eigen vectors of A'*A which is M*M.
[V,D] = eig(A'*A);
V = fliplr(V);
eigval = diag(D);

%% Find eigen vectors of A*A', by A*(eigen vectors of A'*A) => A*V

% Number of vectors with consideration of high energy eigen values
No_Vectors = k; % Number of vectors with consideration of 
eigvec = A*V(:,1:No_Vectors);
eigval = eigval(k:-1:1);

end