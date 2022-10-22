function [UpdateY_P, A,T, obj] = RLRRP(X,class_id,lambda1, lambda2, subspace_c, iters,  knum)
% X:              each column is a data point
% class_id:       a column vector, label vector
% iters:          iteration times
% subspace_c:     dimensionality of subspace
% lambda1:        parameters
% lambda2:        parameters
% knum:           k neighbors

[dim, N] = size(X);
[~, tolN] = size(X);



distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
A = zeros(tolN);  
rr = zeros(tolN,1);
for i = 1:tolN
    di = distX1(i,2:knum+2);
    rr(i) = 0.5*(knum*di(knum+1)-sum(di(1:knum)));   
    id = idx(i,2:knum+2);
    A(i,id) = (di(knum+1)-di)/(knum*di(knum+1)-sum(di(1:knum))+eps);
end
A0 = (A+A')/2;
D = diag(sum(A0));
L = D - A0;
clear A0 D0;
r = mean(rr);


for i = 1 : N
    T(class_id(i),i)  = 1.0;
end

XMean = mean(X,2);
X = X - repmat(XMean, 1, N);
obj = zeros(iters,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UpdateY_P = eye(dim,subspace_c); 
UpdateY_U = eye(dim,dim);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter = 1:iters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Optimize UpdateY_R
    SVD1 = T*X'*UpdateY_P;    
    [U0,~,V0] = svd(SVD1,'econ');
%     [U0,~,V0] = mySVD(SVD1,'econ');    
    UpdateY_R = U0*V0';
    clear U0 V0 SVD1
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Optimize UpdateY_P
    XLXXXU = X * L * X' + lambda1 * X * X' + lambda2 * UpdateY_U;
    XQR = X * T' * UpdateY_R;
    UpdateY_P = lambda1 * (XLXXXU \ XQR);
    

    b = T - UpdateY_R*UpdateY_P' * X;
    b = mean(b,2);    
    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        %update UpdateY_U
    for i=1:dim
        UpdateY_U(i,i) = 1 / (2 * norm(UpdateY_P(i,:),2) + eps) ;
    end
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Optimize matrix T
    U =UpdateY_R * UpdateY_P' * X + b*ones(1,N);
    for ind = 1:N
        T(:,ind) = optimizedT(U(:,ind), class_id(ind));
    end
    clear R U;
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % Optimize UpdateY_A
    distx = L2_distance_1(UpdateY_P'*X,UpdateY_P'*X);
    if iter>1
        [~, idx] = sort(distx,2);
    end
    A = zeros(tolN);
    for j=1:tolN
        idxa0 = idx(j,2:knum+1);
        dxi = distx(j,idxa0);
        ad = -dxi/(2*r);
        A(j,idxa0) = EProjSimplex_new(ad);
    end
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
     
    T1 = UpdateY_R*UpdateY_P' * X + b*ones(1,N)-T;
    T2 = trace(UpdateY_P'*X * L * X' * UpdateY_P) ;%+ r * trace(A' * A);
    T3 = trace(UpdateY_P' * UpdateY_U * UpdateY_P); %+ lambda2 * T3
    obj(iter)=lambda1 * trace(T1'*T1) + T2 + lambda2 * T3; clear T1 T2 T3; 
    clear distx idxa0 dxi ad
end
end

function T = optimizedT(R, label)
classNum = length(R);
T = zeros(classNum,1);
V = R + 1 - repmat(R(label),classNum,1);    
step = 0;
num = 0;
for i = 1:classNum
    if i~=label
        dg = V(i);
        for j = 1:classNum
            if j~=label
                if V(i) < V(j)
                    dg = dg + V(i) - V(j);
                end
            end
        end
        if dg > 0
            step = step + V(i);
            num = num + 1;
        end
    end
end
step = step / (1+num);
for i = 1:classNum
    if i == label
        T(i) = R(i) + step;
    else
        T(i) = R(i) + min(step - V(i), 0);
    end
end
end

function [W, b] = LSR(X, Y, gamma)
[dim, N] = size (X);
[~, N] = size(Y);
XMean = mean(X')';
XX = X - repmat(XMean, 1, N);

W = [];
b = [];
t0 =  XX * XX' + gamma * eye(dim);
W = t0 \ (XX * Y');   
b = Y - W' * X;
b = mean(b')'; 
end
