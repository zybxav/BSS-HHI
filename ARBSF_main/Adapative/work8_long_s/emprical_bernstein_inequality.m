clc;
clear;
syms t n delta_l theta_l m
R=1;
epsilon=0;
p=exp(epsilon)/(exp(epsilon)+1);
variance=power(p*(1-p),1/2);
delta0=delta_l/(3*m);
theta0=theta_l/2;

%t=solve((power(variance*2*log(3/delta)/t,1/2))+3*R*log(3/delta)/t-theta==0)
%n=solve(R*power(log(2/delta)/(2*n),1/2)-theta==0)
n=solve(R*power(log(1/delta0)/(2*n),1/2)-theta0==0,n)
t=solve((power(variance*2*log(1.5/delta0)/t,1/2))+3*R*log(1.5/delta0)/t-theta0==0)


%这边我的选择为啥是1.5而不是3




