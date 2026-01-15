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
t=0.05:(tj-0.05)/(d-1):tj;
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