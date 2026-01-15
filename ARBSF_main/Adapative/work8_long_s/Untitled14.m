clc;
clear all;
syms delta0 result variance0 theta00 delta0 N t_time
t_time = 1;
variance0 = 0;
temp0 = 0;
for theta00 = 0: 0.01:0.5
  temp0 = temp0 + 1;
  result2 =solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta00==0,delta0);
  delta2(temp0) = vpa(1/result2);
end
plot(1:1:temp0, delta2)