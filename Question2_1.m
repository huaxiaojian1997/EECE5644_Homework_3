% Initial parameter setting
clear all, close all,
n = 2; % number of feature dimensions
N = 999; % number of iid samples
mu(:,1) = [0;0]'; mu(:,2) = [2;2]';
Sigma(:,:,1) = [3,0.5; 0.5,5]; Sigma(:,:,2) = [4,-1.9; -1.9,2];
p = [0.3,0.7]; % class priors for labels - and + respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
for l = 0:1 % Draw samples from each class pdf
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

% plot data and their true labels
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class -','Class +'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

% MAP decision
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decisionMAP = (discriminantScore >= log(gamma));
ind00 = find(decisionMAP==0 & label==0); % probability of true negative
ind10 = find(decisionMAP==1 & label==0); % probability of false positive
ind01 = find(decisionMAP==0 & label==1); % probability of false negative
ind11 = find(decisionMAP==1 & label==1); % probability of true positive
errorMAP = length(ind10) + length(ind01); % the number of error
proberrorMAP = errorMAP/N; % probability of error, empirically estimated
fprintf('Number of  error counts for MAP classifier is %d',errorMAP);
fprintf('\n');
fprintf('Probability of error for MAP classifier is %.4f',proberrorMAP);
fprintf('\n');

% plot data and their inferred (decision) labels by MAP classifier
figure(2), clf,% class - circle, class + plus sign, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data and their inferred (decision) labels by MAP classifier'),
xlabel('x_1'), ylabel('x_2'), 

clear ind00 ind01 ind10 ind11

% plot LDA projection of data and their true labels
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
figure(3), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class -','Class +'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

% Estimate the ROC curve for this LDA classifier
[ROCLDA,tauLDA] = estimateROC(yLDA,label);
probErrorLDA = [ROCLDA(1,:)',1-ROCLDA(2,:)']*[sum(label==0),sum(label==1)]'/N; % probability of error for LDA for different threshold values
pEminLDA = min(probErrorLDA);   % minimum probability of error
ind = find(probErrorLDA == pEminLDA);

% LDA deciosion
decision = (yLDA >= tauLDA(ind(1))); % use smallest min-error threshold
ind00 = find(decision==0 & label==0); % probability of true negative
ind10 = find(decision==1 & label==0); % probability of false positive
ind01 = find(decision==0 & label==1); % probability of false negative
ind11 = find(decision==1 & label==1); % probability of true positive
errorLDA = length(ind10) + length(ind01); % the number of error
proberrorLDA = errorLDA/N; % probability of error, empirically estimated
fprintf('Number of  error counts for nLDA classifier is %d',errorLDA);
fprintf('\n');
fprintf('Probability of error for nLDA classifier is %.4f',proberrorLDA);
fprintf('\n');

% plot LDA projection of Data and their inferred (decision) labels
figure(4), clf,% class - circle, class + +, correct green, incorrect red
plot(yLDA(1,ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(1,ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(1,ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(1,ind11),zeros(1,length(ind11)),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('LDA projection of Data and their inferred (decision) labels'),
xlabel('x_1'), ylabel('x_2'),

% plot data and their inferred (decision) labels by LDA classifier
figure(5), clf,
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data and their inferred (decision) labels by LDA classifier'),
xlabel('x_1'), ylabel('x_2'),

clear ind00 ind01 ind10 ind11

% LDA deciosion
% decisionLDA = (yLDA >= 0);
% ind00 = find(decisionLDA==0 & label==0); % probability of true negative
% ind10 = find(decisionLDA==1 & label==0); % probability of false positive
% ind01 = find(decisionLDA==0 & label==1); % probability of false negative
% ind11 = find(decisionLDA==1 & label==1); % probability of true positive
% errorLDA = length(ind10) + length(ind01); % the number of error
% proberrorLDA = errorLDA/N; % probability of error, empirically estimated
% fprintf('Number of  error counts for LDA classifier is %d',errorLDA);
% fprintf('\n');
% fprintf('Probability of error for LDA classifier is %.4f',proberrorLDA);
% fprintf('\n');

% plot data and their inferred (decision) labels by LDA classifier
% figure(6), clf,% class - circle, class + +, correct green, incorrect red
% plot(x(1,ind00),x(2,ind00),'og'); hold on,
% plot(x(1,ind10),x(2,ind10),'or'); hold on,
% plot(x(1,ind01),x(2,ind01),'+r'); hold on,
% plot(x(1,ind11),x(2,ind11),'+g'); hold on,axis equal,
% legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
% title('Data and their inferred (decision) labels by LDA classifier'),
% xlabel('x_1'), ylabel('x_2'),
% 
% clear ind00 ind01 ind10 ind11

%  logistic-linear classifier
bLDA = tauLDA(ind(1)); % smallest min-error threshold
% fun1 = sum(log(1-1./(1+exp([a(1) a(2)]*x(:,find(L==1))+a(3)))));
% fun2 = sum(log(1./(1+exp([a(1) a(2)]*x(:,find(L==2))+a(3)))));
% fun = @(a)(sum(1-1./(1+exp([a(1) a(2)]*x(:,find(label==0))+a(3))))+sum(1./(1+exp([a(1) a(2)]*x(:,find(label==1))+a(3)))))/N; 
fun = @(a)(sum(log(1-1./(1+exp([a(1) a(2)]*x(:,find(label==0))+a(3)))))+sum(log(1./(1+exp([a(1) a(2)]*x(:,find(label==1))+a(3))))))/N*(-1); 
b = [wLDA' bLDA]; 
a = fminsearch(fun,b);
decisionLL = (1./(1+exp([a(1) a(2)]*x+a(3))) >= (1-1./(1+exp([a(1) a(2)]*x+a(3)))));
ind00 = find(decisionLL==0 & label==0); % probability of true negative
ind10 = find(decisionLL==1 & label==0); % probability of false positive
ind01 = find(decisionLL==0 & label==1); % probability of false negative
ind11 = find(decisionLL==1 & label==1); % probability of true positive
errorLL = length(ind10) + length(ind01); % the number of error
proberrorLL = errorLL/N; % probability of error, empirically estimated
fprintf('Number of  error counts for logistic-linear classifier is %d',errorLL);
fprintf('\n');
fprintf('Probability of error for logistic-linear classifier is %.4f',proberrorLL);
fprintf('\n');

% plot Data and their inferred (decision) labels by logistic-linear classifier
figure(6), clf,% class - circle, class + +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data and their inferred (decision) labels by logistic-linear classifier'),
xlabel('x_1'), ylabel('x_2'),

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [ROC,tau] = estimateROC(yLDA,label)
% Generate ROC curve samples
Nc = [length(find(label==0)),length(find(label==1))];
sortedScore = sort(yLDA,'ascend');
tau = [sortedScore(1)-1,(sortedScore(2:end)+sortedScore(1:end-1))/2,sortedScore(end)+1];
% thresholds at midpoints of consecutive scores in sorted list
for k = 1:length(tau)
    decision = (yLDA >= tau(k));
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
    ROC(:,k) = [p10;p11];
end
end