clc;
clear all;

dbstop if error
%input


clear all;
clc;
d=100;
N=10000;
t=zeros(1,d);

tj=1;
%y = normpdf(x,mu,sigma) 返回具有均值 mu 和标准差 sigma 的正态分布的 pdf，在 x 中的值处计算函数值。
x = 0.1:0.1:10;
y = normpdf(x,5,1/sqrt(2*pi));
y=y;
t=y;
%设计成高斯分布

data=zeros(N,d);
for I=1:N
    for k=1:d
         temp1=rand(1,1);
            temp2=t(1,k);
            if  temp1<=temp2
               data(I,k)=1; 
            else
                 data(I,k)=0; 
            end
    end
end
Ground_truth=sum(data);
AAA=data;
writematrix(AAA,'data.csv');
%写一个非binary的记录。
LL=zeros(1,N);
for gd=1:N
    LL=length(find(AAA(gd,:))~=0);
end
DDD=zeros(N,LL);
for ih=1:N
  temp=find(AAA(ih,:)~=0);
  temp2=length(temp);
  DDD(ih,1:temp2)=temp;
end
writematrix(AAA,'data_value.csv');