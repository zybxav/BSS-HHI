syms delta0 result variance0 theta00 delta0 N t
%variance0 =0.17546765026911063;
N=50
t=50 %1的个数，t<N
t_time = N;
n1 = t/(N-1)
n2 = (t+1)/(N)
delta_theta00 = abs(n2 - n1)
%variance0 是由t和N来决定
tp=zeros(N,1);
tp(1:t+1)=1;
variance0 = var(tp)

result1=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-delta_theta00==0,delta0);
vpa(result1)

theta00 = 0.1;
result1=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta00==0,delta0);
vpa(result2)

% 现在因为 theta00 很小，所以
theta00 = 0.1+delta_theta00;
result2=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta00==0,delta0);
vpa(result2)

d1= vpa(1/result1)
d2= vpa(1/result2)

% 多久更新一次，就是所有数据几乎都被选到了？  什么叫做值得更新。 那就是 中间的数据总统计数目为多少？
% 这种

abs(d1-d2)
% 上下各加10 1
% 算出来占总资源的变化？

