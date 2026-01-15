clear all;

h=1:10:200;
epsilon=3;

e=exp(epsilon);
p=e./(1+e);
q=1-p;
theta=p-q;
d=10;
%h就是n0


temp=d*exp(-power(theta,2)*h/(2*d));

y = zeros(1,length(h))+1;   % 定义纵坐标范围

plot(h, y, '^r-')  % 使用红色实线绘制线条
%
hold on;
plot(h,temp,'sb-')