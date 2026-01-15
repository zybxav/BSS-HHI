clc;
clear all;
syms  h N d theta
N=60000;
d=100;
ep=exp(0.5);
theta=1;
T=(N-h*d)*(1-d*exp(-power(theta,2)*h/2));
dy=diff(T,h);
h=solve(dy);
h=vpa(h)
