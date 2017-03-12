%% Principal Component Analysis Based On Expectation Maximization Algorithm
function [W,sigma_square,Xn,t_mean] = em_ppca(t,k)

% Check dimenality Constraint
if k<1 |  k>size(t,2)
   error('Number of PCs must be integer, >0, <dim');
end

%No of Iteration
iter = 20;

% find height and width of data matrix
[height,width] = size(t); 

% Find mean value of [t1 t2 ... tm]
% Mind mean vector of observed data vectors
t_mean = (sum(t(:,1:width)')')/width; 

% Normalize Data matrix
t = t - (t_mean)*ones(1,width);

% Initialy W and sigma^2 will be randomly selected.
W = randn(height,k);
sigma_square = randn(1,1);

disp('EM Algorithm running...');
for i=1:iter
    % Find M = W'W + Sigma^2*I
    M = W'*W + sigma_square*eye(k,k); 
    
    % Find inverse of M
    In_M = inv(M);     
    
    % Expected Xn
    x = (In_M)*W'*t; 
    
    % Find Expected of XnXn'
    xn_xn_T = sigma_square*(In_M) + x*x'; 
    
    % Take Old value of W
    old_W = W;
    
    % Take new value of W
    W = t*x'*inv(xn_xn_T); 
    
    sigma_square = 0;
    for i=1:width
        sigma_square=sigma_square + (norm(t(:,1)-t_mean)^2 - 2*((In_M)*(old_W')*(t(:,i)-t_mean))'*(W')*(t(:,i)-t_mean));
    end
    sigma_square = sigma_square + trace(xn_xn_T*(W')*W);
    
    sigma_square = sigma_square/(width*height);
end

disp('EM Algorithm completed. W is created. Press enter to continue...');
pause;

% Find M = W'W + sigma^2*I
M = W'*W + sigma_square * eye(k,k);
% Inverse of M
In_M = inv(M); 
% Latent Variable Xn
Xn = W'*t; 
disp('Principal Component are ready...press enter to continue...');
pause;

end