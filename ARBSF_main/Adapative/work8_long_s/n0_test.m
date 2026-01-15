N=10000;
d=100;
epsilon=10;
syms  h theta
%这里的theta是啥,默认是1吗？
%为什么和epsilon无关，我记得是越小关系越大拉着
theta=1;
e=exp(epsilon);
p=e/(1+e);
q=1-p; 
theta=p-q;
T=(N-h*d)*(1-d*exp(-power(theta,2)*h/(2*d)));
dy=diff(T,h);
h=solve(dy);
h=ceil(vpa(h))