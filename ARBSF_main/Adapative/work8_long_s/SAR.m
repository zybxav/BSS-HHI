%SR算法
clc;
clear;
% 设置参数的值
K = 20;   % 设置机器的个数
n = 10000;     % 设置总共迭代的次数,所有的次数相加不会超过这个值
X = [zeros(3,K);ones(1,K)];
X = X';

% 计算LogK
LogK = 1/2;
for i= 2:K
    LogK = LogK + 1/i;
end

% 计算每次应该抽取的次数N
N = zeros(1,K);
for k = 2:K
    N(k) = (n-K)/(LogK*(K+1-k));
end
for k = 1:K-1
    N(k) = N(k+1)-N(k);
end

for m = 1:K-1
    disp(m);   % 输出在第几轮
for i=1:N(m)
for j = 1:K
    if(X(j,4)~=0)
    X(j,1) = X(j,1)+normrnd(1.5+0.5*j,10);
    X(j,2) = X(j,2)+1;
    X(j,3) = X(j,1)/X(j,2);
    end
end
end
   [~,indexJ] = min(X(:,3));
    X(indexJ,4)=0;
    X(indexJ,3)=inf;
end

%% 计算完毕之后，对于结果一次性进行处理
for i = 1:K
    X(i,3) = X(i,1)/X(i,2); 
end

%% 输出结果
J = find(X(:,4)~=0);
disp(J);
