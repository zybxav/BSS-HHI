%SR算法
%现在的整个版本是没有和数据线管的。和数据本身和theta完全无关，所以我觉得那个推导是存在问题的
%而且我认为不同的轮数结果是不同的
%非LDP扰动版本
clc;
clear;
% 设置参数的值
n = 40;   % 设置机器的个数
m=2;  %这是个啥
t=zeros(2,n);
t(1,:)=0:1/(n):1-1/n;
t(2,:)=zeros(1,n)+1;
l=1;
theta0=0.05;
delta0=0.01;
theta(1)=theta0;
delta(1)=delta0/2;

H1=ceil(log2(n/m));
T=zeros(n,H1+1);
for o=1:n;
    T(o,1)=o;
end
nl=n;
for l=1:H1  
    nl=ceil(nl/2);
    for k=1:n
        %纯bandit的写法
          if  t(2,k)==1
        Hk=ceil(2/(power(theta(l),2))*log(3*m/delta(l)))
        
          temp3=0;
          for i=1:Hk
            temp1=rand(1,1);
            temp2=t(1,k);
            if  temp1<=temp2
               temp3=temp3+1; 
            end   
          end
          temp4=temp3/Hk;
          T(k,l+1)=temp4;
           
          else
         end
    end
    T0=sortrows(T,l+1);
   if nl>=m
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
end