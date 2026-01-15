%SR算法
%现在的整个版本是没有和数据线管的。和数据本身和theta完全无关，所以我觉得那个推导是存在问题的
%而且我认为不同的轮数结果是不同的
%这个版本要添加扰动
%非LDP扰动版本
%要添加扰动阶段
%版本一，直接套原始的定义
%版本二，和数据方差相关，根据方差挑选数目
%和数据的本身的差距相关，由此挑选数据

clc;
clear;
% 设置参数的值
n = 40;   % 设置机器的个数
m=10;  %这是个啥
N=50000;
theta0=0.5;
delta0=0.1;
Dataset_generator
t=zeros(2,n);
t(1,:)=0:1/(n):1-1/n;
t(2,:)=zeros(1,n)+1;
epsilon=1;
e=exp(epsilon);
p=e/(e+1);
q=1/(e+1);

theta(1)=theta0;
delta(1)=delta0/2;


H1=ceil(log2(n/m));
T=zeros(n,H1+1);
for o=1:n;
    T(o,1)=o;
end
l=1;

Procedure_matrix=zeros(n,2);
Uo=0;
nl=n;
for l=1:H1  
   
    for k=1:nl
        %纯bandit的写法
          if  t(2,k)==1
        sample_k=ceil(2/(power(theta(l),2))*log(3*m/delta(l)));
          temp3=0;
          
          for i=1:sample_k
               Uo=Uo+1;
            temp1=rand(1,1);
            temp2=data(Uo,k);
            if  temp1<=p
               temp3=temp2;
            else
                temp3=1-temp2;
            end  
            if temp3==1
                 Procedure_matrix(k,2)=Procedure_matrix(k,2)+1;
            else
                 Procedure_matrix(k,1)=Procedure_matrix(k,1)+1;
            end
          end
          temp4=Procedure_matrix(k,2);
          %数据恢复
          temp4=(temp4-sample_k*q)/(p-q);
          T(k,l+1)=temp4;
          else
         end
    end
    T0=sortrows(T,l+1);
   if nl>m
        T2=T0(1:n-nl,1);
        T1=T0(nl+1:n,1);
     else
         T2=T0(1:n-m,1);    
   end
        ht=size(T2);
       ht1=ht(1);
       for ij=1:ht1
       t(2,ij)=0;
       end
  %find   
  theta(l+1)=3/4*theta(l);
  delta(l+1)=1/2*delta(l);
   nl=ceil(nl/2);
end