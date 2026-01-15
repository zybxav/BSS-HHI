clc;
clear;
syms t n delta0 theta0 m
R=1;
epsilon=1;
p=exp(epsilon)/(exp(epsilon)+1);
x=0.5; 
C=0.4;
P=p*x+(1-p)*(1-x);
variance=power(P*(1-P),1/2);
%delta0=delta_l/(3*m);
%theta0=theta_l/2;
%theta0=px-C;
%theta0=x-C;

t=100;
%delta0=solve((power(variance*2*log(1.5/delta0)/t,1/2))+3*R*log(1.5/delta0)/t-theta0==0,delta0)
delta0=solve((power(variance*2*log(3/delta0)/t,1/2))+3*R*log(3/delta0)/t-theta0==0,delta0)
%theta0这个错误概率为啥会大于1? 





