clc
clear all
load('Test2.mat')
D=zeros();
H=zeros();
E=string(H);
temp1=0;
F=zeros();
for i=1:200000
    if length(find(strcmp(C(i), E)))>=1
    C(i);
    test1=1;
    F(i)=find(strcmp(C(i), E));
    else
    C(i);
    test2=2;
    temp1=temp1+1;
    if i==1
    E=C(i);
    else
        E=[E,C(i)];
    end
    D=[D,temp1];    
    F(i)=temp1;
    end
    %给当前的数据编码
end
