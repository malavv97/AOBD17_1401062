%% Principal Component Analysis Based On Expectation Maximization Algorithm
function [W,sigma_square,Xn] = ppca(t,k)

% Check dimenality Constraint
if k<1 |  k>size(t,2)
   error('Number of PCs must be integer, >0, <dim');
end

% find height and width of data matrix
[height,width] = size(t); 

% Find mean value of [t1 t2 ... tm]
% Mind mean vector of observed data vectors
t_mean = (sum(t(:,1:width)')')/width; 

% Normalize Data matrix
t = t - (t_mean)*ones(1,width);
S = zeros(height,height);
% Find sample co-variance matrix
for i=1:width
    S = S + (t(:,i)-t_mean)*(t(:,i)-t_mean)';
end
S = S/width; % Sampled covariance matrix

%Find eigen value and eigen vectors
[V,D] = eig(S);
% Extract diagonal components
D = diag(D);
% Sort in descending order for getting higher eigen values
[D, i] = sort(D, 'descend');
% Consider higher eigen values related eigen vectors
V = fliplr(V);
U = V(:,1:k);

lambda_diag = diag(D);
L = lambda_diag(1:k, 1:k);

sigma_square = 1/(d-q) * sum(lambda(k+1:width));
W = U * sqrt(L - sigma_square*eye(k,k));

M = W'*W + sigma_square*eye(k,k);
In_M = inv(M); % Inverse of M
Xn = (In_M)*W'*t; % Latent Variable Xn

end