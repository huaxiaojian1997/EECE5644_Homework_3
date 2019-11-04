close all,
N = 1000; % N samples
L = 6; % True model is of order L
delta = 1e-1;% tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 4-component GMM
alpha_true = [0.25,0.25,0.25,0.25];
mu_true = [-10 0 10 0;0 0 0 10];
Sigma_true(:,:,1) = [3 1;1 20];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 1;1 16];
Sigma_true(:,:,4) = [5 1;1 8];
% Sigma_true(:,:,5) = [6 1;1 6];
% Sigma_true(:,:,6) = [8 1;1 3];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
[d,~] = size(mu_true); % determine dimensionality of samples and number of GMM components

% Divide the data set into 10 approximately-equal-sized partitions
K = 10; % K fold cross validation
dummy = ceil(linspace(0,N,K+1));
indPartitionLimits = zeros(K,2);
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Allocate space
EMtrain = zeros(K,L); EMvalidate = zeros(K,L); 

    % K-fold cross validation
for k = 1:K
    indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
    xValidate = x(:,indValidate); % Using folk k as validation set
    if k == 1
        indTrain = [indPartitionLimits(k,2)+1:N];
    elseif k == K
        indTrain = [1:indPartitionLimits(k,1)-1];
    else
        indTrain = [(1:indPartitionLimits(k-1,2)),(indPartitionLimits(k,2)+1:N)];
    end
    xTrain = x(:,indTrain); % using all other folds as training set
    Ntrain = length(indTrain); Nvalidate = length(indValidate);
    
    % Initialize the GMM to randomly selected samples
    for M = 1:L
        alpha = ones(1,M)/M;
        shuffledIndices = randperm(Ntrain);
        mu = xTrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
        [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); % assign each sample to the nearest mean
        for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
            Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        end
        
        Converged = 0; % Not converged at the beginning
        while ~Converged
            temp = zeros(M,Ntrain);
            for l = 1:M
                temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
            muNew = xTrain*w';
            for l = 1:M
                v = xTrain-repmat(muNew(:,l),1,Ntrain);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
        end
        
        % Train model parameters
        EMtrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)));
        EMvalidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)));
    end
end
AverageEMtrain = mean(EMtrain); % average training EM over 10-folds
AverageEMvalidate = mean(EMvalidate); % average validation EM over 10-folds

fprintf('N = %d', N);
fprintf('\n');
disp('Average training EM over 10-folds is');
disp(AverageEMtrain);
disp('Average validation EM over 10-folds is');
disp(AverageEMvalidate);

figure(1), clf,
Components = 1:L;
plot(Components,AverageEMtrain,'+r'); hold on; 
plot(Components,AverageEMvalidate,'ob');
title(['10-fold Cross-validation of ',num2str(N),' Samples']),
xlabel('Model GMM Orders'); ylabel('EM estimate with 10-fold cross-validation');
legend('Training EM','Validation EM');

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end